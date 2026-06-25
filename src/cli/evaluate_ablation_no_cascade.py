import logging
from pathlib import Path
import torch
import argparse
from src.experiments.ablation_no_cascade.model import cascDP_Ablation1
from src.models.backbone import create_backbone
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
    p = argparse.ArgumentParser(description="Evaluate ablation_no_cascade model (no disorder cascade)")
    p.add_argument("--checkpoint", required=True, help="Path to ablation checkpoint (.pt)")
    p.add_argument(
        "--test-set",
        required=True,
        choices=[
            "caid3_binding",
            "caid3_binding_idr",
            "caid3_linker",
            "test_final",
        ],
        help="Test set",
    )
    p.add_argument("--output-dir", default="results/evaluations/ablation_no_cascade", help="Output directory")
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
        help="Gaussian smoothing sigma applied post-prediction. 0 disables smoothing.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p

def load_model(checkpoint_path: str, device: str = "cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" not in ckpt:
        raise ValueError("Checkpoint missing 'model_config'. Was it saved by AblationTrainer?")

    cfg = ckpt["model_config"]
    arch = cfg.get("architecture_version", "")
    if arch != "ablation_no_cascade":
        raise ValueError(f"Checkpoint architecture_version={arch!r}; expected 'ablation_no_cascade'")

    backbone_type = cfg.get("backbone_type", "esmc")
    if "backbone_name" not in cfg:
        raise ValueError("Checkpoint model_config must contain 'backbone_name'")
    model_name = cfg["backbone_name"]

    # LoRA detection
    state_keys = list(ckpt["model_state_dict"].keys())
    has_lora = any("lora_" in k or "base_model" in k for k in state_keys)

    if has_lora:
        lora_cfg = cfg.get("lora", {})
        lora_r = lora_cfg.get("r", 64)
        lora_alpha = lora_cfg.get("lora_alpha", 128)
        lora_dropout = 0.0          # no dropout at inference
        target_modules = lora_cfg.get(
            "target_modules",
            ["attn.out_proj", "ffn.1", "ffn.3"],
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

    model = cascDP_Ablation1(
        backbone=backbone,
        device=device,
        context_type=cfg.get("phase2_context_type", cfg.get("context_type", "bigru")),
        binding_context_type=cfg.get("binding_context_type"),
        linker_context_type=cfg.get("linker_context_type"),
        use_binding_head=cfg.get("use_binding_head", True),
        use_linker_head=cfg.get("use_linker_head", True),
        freeze_backbone=True,   # always frozen at eval time
        dropout=cfg.get("dropout", 0.5),
        binding_combined=cfg.get("binding_combined", False),
        binding_head_type=cfg.get("binding_head_type", "cnn"),
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    bm = ckpt.get("best_metric", "?")
    bmv = ckpt.get("best_metric_value", ckpt.get("best_val_loss", float("nan")))
    logging.info(f"Loaded ablation checkpoint (epoch {epoch}) — {bm}: {bmv:.4f}")

    saved_thresholds = {
        "binding": ckpt.get("best_binding_threshold", None),
        "linker": ckpt.get("best_linker_threshold",  None),
    }
    saved_thresholds = {
        k: parsed
        for k, v in saved_thresholds.items()
        if (parsed := parse_finite_threshold(v)) is not None
    }
    if saved_thresholds:
        logging.info(
            f"Loaded thresholds from checkpoint: "
            + ", ".join(f"{k}={v:.4f}" for k, v in saved_thresholds.items())
        )

    return model, saved_thresholds

def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = choose_device()
    set_seed(42)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, saved_thresholds = load_model(args.checkpoint, device)

    logger.info("Loading test set: %s", args.test_set)
    # batch_size=1 matches evaluate_caid for accurate per-sequence timing
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

    logger.info("Ablation evaluation complete: %s", output_dir)

if __name__ == "__main__":
    main()
