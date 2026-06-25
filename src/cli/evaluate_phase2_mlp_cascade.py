import logging
from pathlib import Path
import torch
import argparse
from src.experiments.phase2_mlp_cascade.model import cascDP_Phase2_MLPCascade
from src.models.backbone import create_backbone
from src.models.cascDP_phase1 import cascDP_Phase1
from src.models.cascDP_phase1_recycle import cascDP_Phase1Recycle
from src.models.cascDP_phase2 import cascDP_Phase2
from src.cli.evaluate_caid import (
    create_test_dataloader,
    set_seed,
    predict_and_write_submission,
    run_caid_metrics,
    format_caid_metrics_summary,
    write_timings,
    parse_flavors,
    choose_device,
)
from src.evaluation.thresholds import parse_finite_threshold

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate Phase 2 MLP-cascade model (128-dim MLP hidden as cascade signal)"
    )
    p.add_argument("--checkpoint", required=True, help="Path to MLP-cascade checkpoint (.pt)")
    p.add_argument(
        "--test-set",
        required=True,
        choices=[
            "caid3_binding",
            "caid3_binding_idr",
            "caid3_linker",
            "test_final",
        ],
        help="Test set name",
    )
    p.add_argument(
        "--output-dir",
        default="results/evaluations/phase2_mlp_cascade",
        help="Directory for submission files and metrics",
    )
    p.add_argument(
        "--flavors",
        default=None,
        help="Comma-separated flavor filter, e.g. binding,linker",
    )
    p.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip CAID bootstrap metrics for faster local checks",
    )
    p.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma applied post-prediction (0 = disabled)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p

def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'model_config'. Was it saved by the MLP-cascade trainer?"
        )

    model_cfg = checkpoint["model_config"]
    backbone_type = model_cfg.get("backbone_type", "esmc")
    if "backbone_name" not in model_cfg:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")
    model_name = model_cfg["backbone_name"]

    phase1_context_type = model_cfg.get("phase1_context_type", model_cfg.get("context_type"))
    if phase1_context_type is None:
        raise ValueError("Checkpoint model_config must contain 'phase1_context_type' or Phase 1 'context_type'")
    logging.info(f"Phase 1 context_type: {phase1_context_type}")

    # Auto-detect LoRA
    checkpoint_keys = list(checkpoint["model_state_dict"].keys())
    has_lora = any("lora_" in k or "base_model" in k for k in checkpoint_keys)
    if has_lora:
        lora_cfg = model_cfg.get("lora", {})
        lora_r = lora_cfg.get("r", 64)
        lora_alpha = lora_cfg.get("lora_alpha", 128)
        lora_dropout = 0.0  # no dropout at inference
        target_modules = lora_cfg.get(
            "target_modules", ["attn.out_proj", "ffn.1", "ffn.3"]
        )
        layers_to_transform = lora_cfg.get("layers_to_transform", None)
        use_lora = True
        logging.info(f"LoRA detected (r={lora_r}, alpha={lora_alpha})")
    else:
        use_lora = False
        lora_r = lora_alpha = lora_dropout = None
        target_modules = layers_to_transform = None
        logging.info("No LoRA detected in checkpoint")

    backbone = create_backbone(
        backbone_type=backbone_type,
        model_name=model_name,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )

    # Phase 1 CRF detection
    use_crf = any("phase1.crf.transitions" in k for k in checkpoint_keys)
    if use_crf:
        logging.info("Detected Phase 1 CRF in checkpoint")

    use_crf_linker = model_cfg.get("use_crf_linker", False)
    if use_crf_linker:
        logging.info("Using Linker CRF from checkpoint config")

    # Recycled Phase 1 detection
    is_recycled = any("phase1.recycle_proj" in k for k in checkpoint_keys)
    if is_recycled:
        num_recycles = checkpoint.get("model_config", {}).get("num_recycles", 2)
        logging.info(f"Detected recycled Phase 1 (num_recycles={num_recycles})")
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
        )

    use_binding_head = model_cfg.get("use_binding_head", True)
    if "use_linker_head" not in model_cfg:
        raise ValueError("Checkpoint model_config must contain 'use_linker_head'")
    use_linker_head = model_cfg["use_linker_head"]

    # Resolve Phase 2 context types (same helper used by evaluate_caid)
    phase2_context_type, binding_context_type, linker_context_type = (
        cascDP_Phase2.resolve_context_types(model_cfg)
    )

    # Instantiate MLP-cascade variant — cascade_dim from saved model_config
    cascade_dim = model_cfg.get("cascade_dim", 512)
    logging.info(f"cascade_dim: {cascade_dim}")
    model = cascDP_Phase2_MLPCascade(
        phase1_model=phase1_model,
        device=device,
        context_type=phase2_context_type,
        binding_context_type=binding_context_type,
        linker_context_type=linker_context_type,
        use_binding_head=use_binding_head,
        use_linker_head=use_linker_head,
        use_crf_linker=use_crf_linker,
        cascade_dim=cascade_dim,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    bm = checkpoint.get("best_metric", "?")
    bmv = checkpoint.get("best_metric_value", checkpoint.get("best_val_loss", float("nan")))
    logging.info(f"Loaded MLP-cascade checkpoint (epoch {epoch}) — {bm}: {bmv:.4f}")

    saved_thresholds = {
        "binding": checkpoint.get("best_binding_threshold", None),
        "linker": checkpoint.get("best_linker_threshold", None),
    }
    saved_thresholds = {
        k: parsed
        for k, v in saved_thresholds.items()
        if (parsed := parse_finite_threshold(v)) is not None
    }
    if saved_thresholds:
        logging.info(
            "Loaded thresholds from checkpoint: "
            + ", ".join(f"{k}={v:.4f}" for k, v in saved_thresholds.items())
        )

    return model, saved_thresholds

def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"evaluation_mlp_cascade_{args.test_set}.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    device = choose_device()
    set_seed(42)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, saved_thresholds = load_model(args.checkpoint, device)

    logger.info("Loading test set: %s", args.test_set)
    dataloader, _ = create_test_dataloader(args.test_set, batch_size=1, num_workers=0)

    selected_flavors = parse_flavors(args.flavors)
    output_dir = Path(args.output_dir)
    submission_dir = output_dir / "submissions" / args.test_set
    metrics_root = output_dir / "caid_metrics" / args.test_set
    merged_dir = metrics_root / "_merged_predictions"
    reference_dir = metrics_root / "_references"

    logger.info("Writing CAID submission files to %s", submission_dir)
    predictions, timings, produced_flavors = predict_and_write_submission(
        model=model,
        dataloader=dataloader,
        device=device,
        test_set=args.test_set,
        saved_thresholds=saved_thresholds,
        output_dir=submission_dir,
        selected_flavors=selected_flavors,
        gaussian_sigma=args.gaussian_sigma,
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

    logger.info("MLP-cascade evaluation complete: %s", output_dir)

if __name__ == "__main__":
    main()
