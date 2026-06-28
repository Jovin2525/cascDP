from __future__ import annotations
import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set

os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import SubmissionEmbeddingDataset, collate_fn
from src.evaluation.caid_io import (
    SUBMISSION_TASKS,
    safe_filename,
    write_per_protein_caid,
    write_timings,
)
from src.evaluation.thresholds import parse_finite_threshold
from src.models.backbone import create_precomputed_backbone
from src.models.cascDP_phase1 import cascDP_Phase1
from src.models.cascDP_phase1_recycle import cascDP_Phase1Recycle
from src.models.cascDP_phase2 import cascDP_Phase2

logger = logging.getLogger(__name__)
VALID_TASKS = frozenset({"disorder", "binding", "linker"})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CAID submission files from precomputed embeddings")
    parser.add_argument("--checkpoint", required=True, help="Path to the cascDP checkpoint")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Embedding source: directory of per-protein .pt/.npy/.h5 files or a single .h5/.hdf5 container",
    )
    parser.add_argument("--output-dir", default="submission", help="Directory for CAID flavor outputs")
    parser.add_argument(
        "--tasks",
        default="all",
        help=(
            "Comma-separated task outputs to write: disorder,binding,linker, or all. "
            "all writes every valid task produced by the checkpoint; explicit task lists are required."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Prediction batch size")
    parser.add_argument("--threads", type=int, default=1, help="CPU thread limit (1-24)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()

def parse_tasks(raw: str) -> Optional[Set[str]]:
    tasks = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if not tasks or tasks == {"all"}:
        return None
    if "all" in tasks:
        raise ValueError("--tasks cannot combine all with explicit task names")
    unknown = tasks - VALID_TASKS
    if unknown:
        valid = ", ".join(["all", *sorted(VALID_TASKS)])
        raise ValueError(
            f"Unknown --tasks value(s): {', '.join(sorted(unknown))}. "
            f"Valid values: {valid}"
        )
    return tasks

def validate_requested_tasks(requested_tasks: Optional[Set[str]], produced_tasks: Set[str]) -> Set[str]:
    if requested_tasks is None:
        return set(produced_tasks)

    missing = requested_tasks - produced_tasks
    if missing:
        produced = ", ".join(sorted(produced_tasks)) or "none"
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Requested task(s) were not produced: {missing_list}. "
            f"Produced task(s): {produced}. Check that the checkpoint has the requested heads "
            "and finite saved thresholds."
        )
    return set(requested_tasks)

def configure_threads(threads: int) -> None:
    if threads < 1 or threads > 24:
        raise ValueError("--threads must be between 1 and 24")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        logger.debug("Torch interop threads already configured; leaving as-is")

def load_model(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_config")

    model_cfg = checkpoint["model_config"]
    if "backbone_name" not in model_cfg:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")
    model_name = model_cfg["backbone_name"]
    hidden_dim = model_cfg.get("hidden_dim")
    phase1_context_type = model_cfg.get("phase1_context_type", model_cfg.get("context_type"))
    if phase1_context_type is None:
        raise ValueError("Checkpoint model_config must contain 'phase1_context_type' or Phase 1 'context_type'")

    backbone = create_precomputed_backbone(model_name=model_name, hidden_dim=hidden_dim)

    checkpoint_keys = list(checkpoint["model_state_dict"].keys())
    is_phase2 = any("binding_head" in key or "linker_head" in key for key in checkpoint_keys)
    use_crf = any("phase1.crf.transitions" in key for key in checkpoint_keys) if is_phase2 else any(
        "crf.transitions" in key for key in checkpoint_keys
    )

    if is_phase2:
        is_recycled_phase1 = any("phase1.recycle_proj" in key for key in checkpoint_keys)
        if is_recycled_phase1:
            num_recycles = checkpoint.get("model_config", {}).get("num_recycles", 2)
            phase1_model = cascDP_Phase1Recycle(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                use_crf=use_crf,
                num_recycles=num_recycles,
            )
        else:
            phase1_model = cascDP_Phase1(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                use_crf=use_crf,
                fusion_type=model_cfg.get("fusion_type", "sum"),
            )

        use_binding_head = model_cfg.get("use_binding_head", True)
        if "use_linker_head" not in model_cfg:
            raise ValueError("Phase 2 checkpoint model_config must contain 'use_linker_head'")
        use_linker_head = model_cfg["use_linker_head"]
        use_crf_linker = model_cfg.get("use_crf_linker", False)
        phase2_context_type, binding_context_type, linker_context_type = cascDP_Phase2.resolve_context_types(model_cfg)

        model = cascDP_Phase2(
            phase1_model=phase1_model,
            device=device,
            context_type=phase2_context_type,
            binding_context_type=binding_context_type,
            linker_context_type=linker_context_type,
            use_binding_head=use_binding_head,
            use_linker_head=use_linker_head,
            use_crf_linker=use_crf_linker,
            binding_combined=model_cfg.get("binding_combined", False),
            binding_head_type=model_cfg.get("binding_head_type", "cnn"),
        )
    else:
        model = cascDP_Phase1(
            backbone=backbone,
            device=device,
            context_type=phase1_context_type,
            use_crf=use_crf,
            fusion_type=model_cfg.get("fusion_type", "sum"),
        )

    model_keys = set(model.state_dict().keys())
    checkpoint_sd = checkpoint["model_state_dict"]
    filtered_sd = {k: v for k, v in checkpoint_sd.items() if k in model_keys}
    model.load_state_dict(filtered_sd, strict=True)
    model.eval()

    saved_thresholds = {
        "disorder": checkpoint.get("best_threshold", None),
        "binding": checkpoint.get("best_binding_threshold", None),
        "linker": checkpoint.get("best_linker_threshold", None),
    }
    saved_thresholds = {
        k: parsed
        for k, v in saved_thresholds.items()
        if (parsed := parse_finite_threshold(v)) is not None
    }
    return model, saved_thresholds

def threshold(saved_thresholds: Mapping[str, float], key: str) -> Optional[float]:
    return parse_finite_threshold(saved_thresholds.get(key))

def forward_model(model, embeddings: torch.Tensor):
    outputs = model(embeddings=embeddings)
    if isinstance(outputs, tuple):
        disorder_logits, binding_logits, linker_logits = outputs
    else:
        disorder_logits = outputs
        binding_logits = None
        linker_logits = None

    if disorder_logits is not None and disorder_logits.shape[-1] == 2:
        disorder_logits = disorder_logits[..., 1:2] - disorder_logits[..., 0:1]
    if linker_logits is not None and linker_logits.shape[-1] == 2:
        linker_logits = linker_logits[..., 1:2] - linker_logits[..., 0:1]

    return disorder_logits, binding_logits, linker_logits

def probabilities_from_outputs(outputs):
    disorder_logits, binding_logits, linker_logits = outputs
    disorder_probs = None
    if disorder_logits is not None:
        disorder_probs = torch.sigmoid(disorder_logits.squeeze(-1)).cpu().numpy()

    combined_binding = None
    if binding_logits is not None:
        if binding_logits.dim() == 2 or (binding_logits.dim() == 3 and binding_logits.shape[-1] == 1):
            binding_scalar = binding_logits.squeeze(-1) if binding_logits.dim() == 3 else binding_logits
            combined_binding = torch.sigmoid(binding_scalar).cpu().numpy()
        else:
            binding_indiv = torch.sigmoid(binding_logits).cpu().numpy()
            combined_binding = 1.0 - np.prod(1.0 - binding_indiv, axis=-1)

    linker_probs = None
    if linker_logits is not None:
        linker_probs = torch.sigmoid(linker_logits.squeeze(-1)).cpu().numpy()
    return disorder_probs, combined_binding, linker_probs

def build_prediction_record(
    sequence: str,
    disorder_probs: Optional[np.ndarray],
    binding_probs: Optional[np.ndarray],
    linker_probs: Optional[np.ndarray],
    saved_thresholds: Mapping[str, float],
) -> Dict[str, Sequence[float]]:
    seq_len = len(sequence)
    record: Dict[str, Sequence[float]] = {"sequence": sequence}

    if disorder_probs is not None:
        disorder_probs = np.asarray(disorder_probs[:seq_len], dtype=float)
        disorder_thr = threshold(saved_thresholds, "disorder")
        if disorder_thr is not None:
            record["disorder_probs"] = disorder_probs.tolist()
            record["disorder_pred"] = (disorder_probs >= disorder_thr).astype(int).tolist()

    if binding_probs is not None:
        binding_probs = np.asarray(binding_probs[:seq_len], dtype=float)
        binding_thr = threshold(saved_thresholds, "binding")
        if binding_thr is not None:
            record["binding_probs_combined"] = binding_probs.tolist()
            record["binding_pred"] = (binding_probs >= binding_thr).astype(int).tolist()

    if linker_probs is not None:
        linker_probs = np.asarray(linker_probs[:seq_len], dtype=float)
        linker_thr = threshold(saved_thresholds, "linker")
        if linker_thr is not None:
            record["linker_probs"] = linker_probs.tolist()
            record["linker_pred"] = (linker_probs >= linker_thr).astype(int).tolist()

    return record

def write_per_protein_submission(
    predictions: Mapping[str, Mapping[str, Sequence[float]]],
    output_dir: Path,
    allowed_tasks: Set[str],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    for task_name, score_key, state_key in SUBMISSION_TASKS:
        if task_name not in allowed_tasks:
            continue

        task_dir = output_dir / task_name
        if task_dir.exists():
            for stale_path in task_dir.glob("*.caid"):
                stale_path.unlink()

        wrote = 0
        for protein_id in sorted(predictions):
            prediction = predictions[protein_id]
            sequence = prediction.get("sequence")
            scores = prediction.get(score_key)
            states = prediction.get(state_key)
            if not sequence or scores is None or states is None:
                continue
            write_per_protein_caid(
                task_dir / f"{safe_filename(protein_id)}.caid",
                protein_id,
                str(sequence),
                scores,
                states,
            )
            wrote += 1

        if wrote:
            outputs[task_name] = task_dir
        else:
            logger.info("Skipping %s submission directory: no proteins have both scores and states", task_name)

    return outputs

@torch.no_grad()
def predict(
    model, dataloader, device: str, saved_thresholds: Mapping[str, float]
):
    predictions: Dict[str, Dict[str, Sequence[float]]] = {}
    produced_tasks: Set[str] = set()
    timings = []

    for batch in tqdm(dataloader, desc="Predicting"):
        start = time.perf_counter()
        embeddings = batch["embeddings"].to(device)
        disorder_probs, binding_probs, linker_probs = probabilities_from_outputs(forward_model(model, embeddings))
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        for index, protein_id in enumerate(batch["protein_ids"]):
            sequence = batch["sequences"][index]
            predictions[protein_id] = build_prediction_record(
                sequence=sequence,
                disorder_probs=disorder_probs[index] if disorder_probs is not None else None,
                binding_probs=binding_probs[index] if binding_probs is not None else None,
                linker_probs=linker_probs[index] if linker_probs is not None else None,
                saved_thresholds=saved_thresholds,
            )
            if "disorder_probs" in predictions[protein_id]:
                produced_tasks.add("disorder")
            if "binding_probs_combined" in predictions[protein_id]:
                produced_tasks.add("binding")
            if "linker_probs" in predictions[protein_id]:
                produced_tasks.add("linker")
            timings.append((protein_id, elapsed_ms))

    return predictions, timings, produced_tasks

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    configure_threads(args.threads)
    try:
        requested_tasks = parse_tasks(args.tasks)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    device = "cpu"
    logger.info("Using device: %s", device)
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, saved_thresholds = load_model(args.checkpoint, device=device)

    dataset = SubmissionEmbeddingDataset(args.embeddings, args.fasta)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    predictions, timings, produced_tasks = predict(model, dataloader, device=device, saved_thresholds=saved_thresholds)
    try:
        allowed_tasks = validate_requested_tasks(requested_tasks, produced_tasks)
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    output_dir = Path(args.output_dir)
    outputs = write_per_protein_submission(
        predictions,
        output_dir,
        allowed_tasks=allowed_tasks,
    )
    write_timings(output_dir / "timings.csv", timings)
    for task_name, output_path in outputs.items():
        logger.info("Wrote %s submission directory: %s", task_name, output_path)

if __name__ == "__main__":
    main()
