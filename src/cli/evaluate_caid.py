from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from src.data.dataset import DisorderFunctionDataset, OnTheFlyDisorderFunctionDataset, collate_fn
from src.models.backbone import create_backbone
from src.models.cascDP_phase1 import cascDP_Phase1
from src.models.cascDP_phase1_recycle import cascDP_Phase1Recycle
from src.models.cascDP_phase2 import cascDP_Phase2
from src.evaluation.caid_eval import (
    BINDING_TYPES,
    format_caid_metrics_summary,
    run_caid_metrics,
    submission_flavors,
)
from src.evaluation.caid_io import safe_filename, write_per_protein_caid, write_timings
from src.evaluation.thresholds import parse_finite_threshold

logger = logging.getLogger(__name__)

def gaussian_smooth_probs(probs: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply 1-D Gaussian smoothing to probability arrays (post-prediction)
    Smooths each sequence independently along residue axis
    """
    if sigma <= 0:
        return probs
    return np.clip(gaussian_filter1d(probs, sigma=sigma), 0.0, 1.0)

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def create_test_dataloader(test_set: str, batch_size: int = 4, num_workers: int = 0):
    # Map test set names to file paths
    test_set_paths = {
        'test_final': {
            'embedding_dir': 'data/embeddings/test_final_600m',
            'disorder_file': 'data/final_cleaned_dataset/test_final_update_or_caid4_unaltered_data.txt' # test_final_update_or_caid4_unaltered_data.txt
        },
        'caid3_disorder_nox': {
            'embedding_dir': None,
            'disorder_file': 'data/eval/caid3/disorder_nox_caid3.fasta'
        },
        'caid3_disorder_pdb': {
            'embedding_dir': None,
            'disorder_file': 'data/eval/caid3/disorder_pdb_caid3.fasta'
        },
        'caid3_binding': {
            'embedding_dir': None,
            'disorder_file': 'data/eval/caid3/binding_caid3.fasta',
            'target_indices': [0]  # Binding head (already aggregated in model)
        },
        'caid3_binding_idr': {
            'embedding_dir': None,
            'disorder_file': 'data/eval/caid3/binding_idr_caid3.fasta',
            'target_indices': [0]  # Binding head
        },
        'caid3_linker': {
            'embedding_dir': None,
            'disorder_file': 'data/eval/caid3/linker_caid3.fasta',
            'target_indices': [1]  # Linker head
        }
    }

    if test_set not in test_set_paths:
        raise ValueError(f"Unknown test set: {test_set}. Choose from {list(test_set_paths.keys())}")

    paths = test_set_paths[test_set]
    disorder_file = paths['disorder_file']
    embedding_dir = paths.get('embedding_dir')
    target_indices = paths.get('target_indices', None)

    if not Path(disorder_file).exists():
        raise FileNotFoundError(f"Disorder file not found: {disorder_file}")

    # Check if embeddings exist
    use_on_the_fly = True
    if embedding_dir and Path(embedding_dir).exists():
        use_on_the_fly = False

    if use_on_the_fly:
        logging.info(f"Using On-The-Fly dataset for {test_set} (embeddings not found at {embedding_dir})")
        dataset = OnTheFlyDisorderFunctionDataset(
            disorder_file=disorder_file,
            embedding_model=None, # Model manages its own backbone
            device='cpu' #
        )
    else:
        logging.info(f"Using Precomputed Embeddings dataset for {test_set}")
        dataset = DisorderFunctionDataset(
            embedding_dir=embedding_dir,
            disorder_file=disorder_file
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return dataloader, target_indices

def load_model(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config. Please retrain model with updated trainer.")

    logging.info("Loading model config from checkpoint")
    model_cfg = checkpoint['model_config']
    backbone_type = model_cfg.get('backbone_type', 'esmc')
    if 'backbone_name' not in model_cfg:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")
    model_name = model_cfg['backbone_name']

    phase1_context_type = model_cfg.get('phase1_context_type', model_cfg.get('context_type'))
    if phase1_context_type is None:
        raise ValueError("Checkpoint model_config must contain 'phase1_context_type' or Phase 1 'context_type'")

    logging.info(f"Model config: Phase 1 context_type={phase1_context_type}")

    # Auto-detect LoRA from checkpoint keys
    checkpoint_keys = checkpoint['model_state_dict'].keys()
    has_lora_keys = any('lora_' in key or 'base_model' in key for key in checkpoint_keys)

    if has_lora_keys:
        # Try to get LoRA config from model_cfg
        lora_cfg = model_cfg.get('lora', {})
        lora_r = lora_cfg.get('r', 32)
        lora_alpha = lora_cfg.get('lora_alpha', 64)
        target_modules = lora_cfg.get('target_modules', ["attn.out_proj", "ffn.1", "ffn.3"])
        layers_to_transform = lora_cfg.get('layers_to_transform', None)

        logging.info(f"Detected LoRA in checkpoint - loading with rank={lora_r}, alpha={lora_alpha}, "
                     f"layers_to_transform={layers_to_transform}")
        use_lora = True
        lora_dropout = 0.0  # No dropout for eval
    else:
        logging.info("No LoRA detected in checkpoint - loading without LoRA")
        use_lora = False
        lora_r = None
        lora_alpha = None
        lora_dropout = None
        target_modules = None
        layers_to_transform = None

    use_binding_head = model_cfg.get('use_binding_head', True)
    logging.info(f"Using binding head: {use_binding_head}")

    backbone = create_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform
    )

    # Detect model phase from checkpoint keys
    checkpoint_keys = list(checkpoint['model_state_dict'].keys())
    is_phase2 = any('binding_' in key or 'linker_' in key for key in checkpoint_keys)

    if is_phase2:
        logging.info("Detected Phase 2 model (function heads)")

        # Create Phase 1 model (recycled or standard)
        is_recycled_phase1 = any('phase1.recycle_proj' in k for k in checkpoint_keys)
        if is_recycled_phase1:
            num_recycles = checkpoint.get('model_config', {}).get('num_recycles', 2)
            logging.info(f"Detected recycled Phase 1 in checkpoint (num_recycles={num_recycles})")
            phase1_model = cascDP_Phase1Recycle(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                num_recycles=num_recycles,
            )
        else:
            phase1_model = cascDP_Phase1(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                fusion_type=model_cfg.get('fusion_type', 'sum'),
            )

        if 'use_linker_head' not in model_cfg:
            raise ValueError("Phase 2 checkpoint model_config must contain 'use_linker_head'")
        use_linker_head = model_cfg['use_linker_head']

        # Create Phase 2 model
        phase2_context_type, binding_context_type, linker_context_type = (
            cascDP_Phase2.resolve_context_types(model_cfg)
        )

        model = cascDP_Phase2(
            phase1_model=phase1_model,
            device=device,
            context_type=phase2_context_type,
            binding_context_type=binding_context_type,
            linker_context_type=linker_context_type,
            use_binding_head=use_binding_head,
            use_linker_head=use_linker_head,
            binding_combined=model_cfg.get('binding_combined', False),
            binding_head_type=model_cfg.get('binding_head_type', 'cnn'),
        )
    else:
        logging.info("Detected Phase 1 model (disorder only)")
        is_recycled_phase1 = any('recycle_proj' in k for k in checkpoint_keys)
        if is_recycled_phase1:
            num_recycles = checkpoint.get('model_config', {}).get('num_recycles', 2)
            logging.info(f"Detected recycled Phase 1 checkpoint (num_recycles={num_recycles})")
            model = cascDP_Phase1Recycle(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                num_recycles=num_recycles,
            )
        else:
            model = cascDP_Phase1(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                fusion_type=model_cfg.get('fusion_type', 'sum'),
            )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Log actual learned bias values (handle Phase 1 vs Phase 2)
    def safe_get_bias(layer):
        if not hasattr(layer, 'bias') or layer.bias is None: return None
        if layer.bias.numel() == 1: return layer.bias.item()
        return layer.bias.detach().cpu().numpy()

    def format_bias_val(val):
        if val is None: return "None"
        if isinstance(val, (float, int)): return f"{val:.4f}"
        import numpy as np
        if isinstance(val, np.ndarray):
             return "[" + ", ".join([f"{x:.4f}" for x in val.flatten()]) + "]"
        return str(val)

    if hasattr(model, 'phase1'):
        # Phase 2 model
        disorder_bias = safe_get_bias(model.phase1.disorder_initial)
        binding_layer = getattr(model, 'binding_output_layer', None)
        if binding_layer is not None and binding_layer.bias is not None:
            binding_bias = binding_layer.bias.detach().cpu().numpy()
        else:
            binding_bias = None

        linker_bias = None
        if hasattr(model, 'linker_head') and hasattr(model.linker_head, 'final'):
             linker_bias = safe_get_bias(model.linker_head.final)
    else:
        # Phase 1 model
        disorder_bias = safe_get_bias(model.disorder_initial)
        binding_bias = None
        linker_bias = None

    disorder_str = format_bias_val(disorder_bias)
    linker_str = format_bias_val(linker_bias)

    if binding_bias is not None:
        binding_types = ['Protein', 'Nucleic', 'Ion', 'Lipid']
        binding_bias_str = ', '.join([f"{t}: {b:.4f}" for t, b in zip(binding_types, binding_bias)])
        logging.info(f"Loaded learned biases - Disorder: {disorder_str}, Binding: [{binding_bias_str}], Linker: {linker_str}")
    else:
        logging.info(f"Loaded learned biases - Disorder: {disorder_str}, Binding: None, Linker: {linker_str}")

    logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    if 'best_metric' in checkpoint:
        best_metric_value = checkpoint.get('best_metric_value', checkpoint.get('best_val_loss', float('nan')))
        logging.info(f"Best {checkpoint['best_metric']}: {best_metric_value:.4f}")

    # Load saved thresholds from checkpoint
    saved_thresholds = {
        'disorder': checkpoint.get('best_threshold', None),
        'binding': checkpoint.get('best_binding_threshold', None),
        'linker': checkpoint.get('best_linker_threshold', None),
    }
    # Drop absent or non-finite thresholds. Do not keep placeholder values such as NaN.
    saved_thresholds = {
        k: parsed
        for k, v in saved_thresholds.items()
        if (parsed := parse_finite_threshold(v)) is not None
    }
    if saved_thresholds:
        logging.info(f"Loaded saved thresholds from checkpoint: { {k: f'{v:.4f}' for k, v in saved_thresholds.items()} }")

    return model, saved_thresholds

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAID-first cascDP evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--test-set",
        required=True,
        choices=[
            "test_final",
            "caid3_disorder_nox",
            "caid3_disorder_pdb",
            "caid3_binding",
            "caid3_binding_idr",
            "caid3_linker",
        ],
        help="Evaluation set",
    )
    parser.add_argument("--output-dir", default="results/evaluations/cascdp", help="Output directory")
    parser.add_argument(
        "--flavors",
        default=None,
        help="Comma-separated flavor filter, e.g. disorder,binding,linker",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip CAID bootstrap metrics for faster local checks",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma for disorder probabilities. 0 disables smoothing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--binding-head",
        default=None,
        choices=["protein", "nucleic", "ion", "lipid"],
        help=(
            "For multi-label (4-output) checkpoints: use this per-type head's probability "
            "as the binding score instead of the noisy-OR combined score. "
            "E.g. --binding-head protein evaluates how well the Protein binding head ranks residues. "
            "Has no effect on combined-head (output_dim=1) checkpoints."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    log_path = f"evaluation_{args.test_set}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )

    device = choose_device()
    set_seed(42)

    logger.info("Loading model from %s", args.checkpoint)
    model, saved_thresholds = load_model(args.checkpoint, device)

    logger.info("Loading test set: %s", args.test_set)
    # Timing is per sequence, so batch size 1
    dataloader, _ = create_test_dataloader(args.test_set, batch_size=1, num_workers=0)

    selected_flavors = parse_flavors(args.flavors)
    output_dir = Path(args.output_dir)
    submission_dir = output_dir / "submissions" / args.test_set
    metrics_root = output_dir / "caid_metrics" / args.test_set
    merged_dir = metrics_root / "_merged_predictions"
    reference_dir = metrics_root / "_references"

    logger.info("Writing official CAID submission files to %s", submission_dir)
    predictions, timings, produced_flavors = predict_and_write_submission(
        model=model,
        dataloader=dataloader,
        device=device,
        test_set=args.test_set,
        saved_thresholds=saved_thresholds,
        output_dir=submission_dir,
        selected_flavors=selected_flavors,
        gaussian_sigma=args.gaussian_sigma,
        binding_head=args.binding_head,
    )
    write_timings(submission_dir / "timings.csv", timings)

    logger.info("Running official CAID metrics")
    completed_metrics = run_caid_metrics(
        dataset=dataloader.dataset,
        test_set=args.test_set,
        submission_dir=submission_dir,
        metrics_root=metrics_root,
        merged_dir=merged_dir,
        reference_dir=reference_dir,
        selected_flavors=selected_flavors,
        produced_flavors=produced_flavors,
        skip_bootstrap=args.skip_bootstrap,
    )
    summary = format_caid_metrics_summary(completed_metrics)
    if summary:
        logger.info("\n%s", summary)

    logger.info("CAID-first evaluation complete: %s", output_dir)

def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def parse_flavors(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    flavors = {item.strip() for item in raw.split(",") if item.strip()}
    return flavors or None

@torch.no_grad()
def predict_and_write_submission(
    model,
    dataloader,
    device: str,
    test_set: str,
    saved_thresholds: Mapping[str, float],
    output_dir: Path,
    selected_flavors: Optional[Set[str]],
    gaussian_sigma: float,
    binding_head: Optional[str] = None,
):
    model.eval()
    seq_lookup = getattr(dataloader.dataset, "sequences", {}) or {}
    predictions: Dict[str, Dict[str, Sequence[float]]] = {}
    produced_flavors: Set[str] = set()
    timings = []

    for batch in tqdm(dataloader, desc="Predicting"):
        protein_ids = batch["protein_ids"]
        pid = protein_ids[0]
        sequence = seq_lookup.get(pid, batch.get("sequences", [""])[0])
        mask = batch["mask"].to(device)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = forward_model(model, batch, device)
        probs = probabilities_from_outputs(outputs, mask, gaussian_sigma, binding_head=binding_head)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        seq_len = min(len(sequence), mask.shape[1])
        pred = build_prediction_record(probs, seq_len, saved_thresholds)
        predictions[pid] = pred

        for flavor, score_key, state_key in submission_flavors(test_set, selected_flavors):
            if score_key not in pred or state_key not in pred:
                continue
            write_per_protein_caid(
                output_dir / flavor / f"{safe_filename(pid)}.caid",
                pid,
                sequence[:seq_len],
                pred[score_key],
                pred[state_key],
            )
            produced_flavors.add(flavor)

        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        timings.append((pid, elapsed_ms))

    return predictions, timings, produced_flavors

def forward_model(model, batch, device: str):
    if "sequences" in batch:
        outputs = model(embeddings=None, sequences=batch["sequences"])
    else:
        outputs = model(batch["embeddings"].to(device))

    if isinstance(outputs, tuple):
        disorder_logits, binding_logits, linker_logits = outputs
    else:
        disorder_logits = outputs
        binding_logits = None
        linker_logits = None

    return disorder_logits, binding_logits, linker_logits

def probabilities_from_outputs(outputs, mask, gaussian_sigma: float, binding_head: Optional[str] = None):
    disorder_logits, binding_logits, linker_logits = outputs

    disorder_probs = None
    if disorder_logits is not None:
        disorder_probs = torch.sigmoid(disorder_logits.squeeze(-1)).cpu().numpy()

    if disorder_probs is not None and gaussian_sigma > 0:
        for i in range(disorder_probs.shape[0]):
            seq_len = int(mask[i].sum().item())
            disorder_probs[i, :seq_len] = gaussian_smooth_probs(disorder_probs[i, :seq_len], gaussian_sigma)

    result = {}
    if disorder_probs is not None:
        result["disorder_probs"] = disorder_probs[0]

    if linker_logits is not None:
        linker_probs = torch.sigmoid(linker_logits.squeeze(-1)).cpu().numpy()
        result["linker_probs"] = linker_probs[0]

    # Branch on output dim: 1 = combined head, 4 = multi-label head
    # CNNHead(output_dim=1) squeezes to (B,L); CNNHead(output_dim=4) keeps (B,L,4)
    if binding_logits is not None:
        if binding_logits.dim() == 2 or (binding_logits.dim() == 3 and binding_logits.shape[-1] == 1):
            scalar_logits = binding_logits.squeeze(-1) if binding_logits.dim() == 3 else binding_logits
            result["binding_probs_combined"] = torch.sigmoid(scalar_logits).cpu().numpy()[0]
        else:
            binding_indiv = torch.sigmoid(binding_logits).cpu().numpy()  # (B, L, 4)
            combined = 1.0 - np.prod(1.0 - binding_indiv, axis=-1)[0]
            for type_name, _label, type_idx in BINDING_TYPES:
                result[f"binding_probs_{type_name}"] = binding_indiv[0, :, type_idx]
            if binding_head is not None and f"binding_probs_{binding_head}" in result:
                result["binding_probs_combined"] = result[f"binding_probs_{binding_head}"]
                logger.debug("Using binding head '%s' as combined binding score", binding_head)
            else:
                result["binding_probs_combined"] = combined

    return result

def build_prediction_record(probs, seq_len: int, saved_thresholds: Mapping[str, float]):
    record = {}
    if "disorder_probs" in probs:
        disorder_probs = np.asarray(probs["disorder_probs"][:seq_len], dtype=float)
        disorder_thr = threshold(saved_thresholds, "disorder")
        if disorder_thr is not None:
            record["disorder_probs"] = disorder_probs.tolist()
            record["disorder_pred"] = (disorder_probs >= disorder_thr).astype(int).tolist()

    if "binding_probs_combined" in probs:
        binding_probs = np.asarray(probs["binding_probs_combined"][:seq_len], dtype=float)
        binding_thr = threshold(saved_thresholds, "binding")
        if binding_thr is not None:
            record["binding_probs_combined"] = binding_probs.tolist()
            record["binding_pred"] = (binding_probs >= binding_thr).astype(int).tolist()

    if "linker_probs" in probs:
        linker_probs = np.asarray(probs["linker_probs"][:seq_len], dtype=float)
        linker_thr = threshold(saved_thresholds, "linker")
        if linker_thr is not None:
            record["linker_probs"] = linker_probs.tolist()
            record["linker_pred"] = (linker_probs >= linker_thr).astype(int).tolist()

    # Emit per-type fields only for multi-label checkpoints (4 outputs)
    for type_name, _label, _type_idx in BINDING_TYPES:
        key = f"binding_probs_{type_name}"
        if key in probs:
            type_probs = np.asarray(probs[key][:seq_len], dtype=float)
            binding_thr = threshold(saved_thresholds, "binding")
            if binding_thr is not None:
                record[key] = type_probs.tolist()
                record[f"binding_{type_name}_pred"] = (type_probs >= binding_thr).astype(int).tolist()
    return record

def threshold(saved_thresholds: Mapping[str, float], key: str) -> Optional[float]:
    return parse_finite_threshold(saved_thresholds.get(key))

if __name__ == "__main__":
    main()
