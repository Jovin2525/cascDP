import os
import math
import random
import logging
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.experiments.phase2_mlp_cascade.model import cascDP_Phase2_MLPCascade
from src.experiments.phase2_mlp_cascade.trainer import MLPCascadeTrainer
from src.models.backbone import create_backbone
from src.models import cascDP_Phase1, cascDP_Phase1Recycle
from src.data.dataset import (
    DisorderFunctionDataset,
    OnTheFlyDisorderFunctionDataset,
    collate_fn,
)
from src.training.loss import CascadedLoss, compute_class_weights

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict, backbone=None):
    logger = logging.getLogger(__name__)
    mode = config["data"].get("embedding_mode", "precomputed")
    logger.info(f"Embedding mode: {mode}")

    train_file = config["data"]["train_disorder_file"]
    val_file = config["data"]["val_disorder_file"]
    pdb_weight = config["training"].get("pdb_loss_weight", 1.0)

    if mode == "precomputed":
        train_ds = DisorderFunctionDataset(
            embedding_dir=config["data"]["train_embedding_dir"],
            disorder_file=train_file,
            pdb_loss_weight=pdb_weight,
        )
        val_ds = DisorderFunctionDataset(
            embedding_dir=config["data"]["val_embedding_dir"],
            disorder_file=val_file,
            pdb_loss_weight=1.0,
        )
    elif mode == "on_the_fly":
        if backbone is None:
            raise ValueError("on_the_fly mode requires backbone model")
        train_ds = OnTheFlyDisorderFunctionDataset(
            disorder_file=train_file,
            embedding_model=backbone,
            device=backbone.device,
            pdb_loss_weight=pdb_weight,
        )
        val_ds = OnTheFlyDisorderFunctionDataset(
            disorder_file=val_file,
            embedding_model=backbone,
            device=backbone.device,
            pdb_loss_weight=1.0,
        )
    else:
        raise ValueError(f"Unknown embedding_mode: {mode}")

    effective_num_workers = (
        0 if mode == "on_the_fly" else config["training"].get("num_workers", 0)
    )
    use_pin_memory = effective_num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader


def create_model(config: dict, device: str) -> cascDP_Phase2_MLPCascade:
    logger = logging.getLogger(__name__)
    lora_cfg = config["model"].get("lora", {})

    backbone = create_backbone(
        backbone_type=config["model"]["backbone_type"],
        model_name=config["model"]["backbone_name"],
        device=device,
        use_lora=config["model"].get("use_lora", False),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        layers_to_transform=lora_cfg.get("layers_to_transform", None),
        target_modules=lora_cfg.get("target_modules", None),
    )

    phase1_checkpoint = config["model"].get("phase1_checkpoint")

    checkpoint = None
    if phase1_checkpoint:
        logger.info(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
        checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)

    phase1_context_type = config["model"].get("phase1_context_type", config["model"].get("context_type", "bigru"))
    if checkpoint and "model_config" in checkpoint:
        ckpt_ctx = checkpoint["model_config"].get("phase1_context_type", checkpoint["model_config"].get("context_type"))
        if ckpt_ctx:
            phase1_context_type = ckpt_ctx
            logger.info(f"Phase 1 context_type from checkpoint: {phase1_context_type}")
    else:
        logger.info(f"Phase 1 context_type from config: {phase1_context_type}")

    # Detect recycled Phase 1
    is_recycled = checkpoint is not None and any(
        "recycle_proj" in k for k in checkpoint["model_state_dict"].keys()
    )
    if is_recycled:
        num_recycles = checkpoint.get("model_config", {}).get("num_recycles", 2)
        logger.info(f"Detected recycled Phase 1 checkpoint (num_recycles={num_recycles})")
        phase1_model = cascDP_Phase1Recycle(
            backbone=backbone,
            device=device,
            context_type=phase1_context_type,
            dropout=config["model"].get("dropout", 0.5),
            num_recycles=num_recycles,
        )
    else:
        phase1_model = cascDP_Phase1(
            backbone=backbone,
            device=device,
            context_type=phase1_context_type,
            dropout=config["model"].get("dropout", 0.5),
        )

    if checkpoint:
        phase1_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.info("Phase 1 checkpoint loaded successfully")
    else:
        logger.warning("No phase1_checkpoint specified — Phase 1 weights are random")

    model = cascDP_Phase2_MLPCascade(
        phase1_model=phase1_model,
        device=device,
        context_type=config["model"].get("phase2_context_type", "bigru"),
        dropout=config["model"].get("dropout", 0.2),
        use_binding_head=config["model"].get("use_binding_head", True),
        use_linker_head=config["model"].get("use_linker_head", True),
        cascade_dim=config["model"].get("cascade_dim", 512),
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 2 MLP-cascade model (128-dim MLP hidden as cascade signal)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.log_level == "DEBUG" else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training_mlp_cascade.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    set_seed(42)
    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info("=== Creating model ===")
    model = create_model(config, device)

    backbone_for_embedding = None
    if hasattr(model, "phase1") and hasattr(model.phase1, "backbone_wrapper"):
        backbone_for_embedding = model.phase1.backbone_wrapper

    logger.info("=== Loading data ===")
    train_loader, val_loader = create_dataloaders(
        config,
        backbone=backbone_for_embedding
        if config["data"].get("embedding_mode") == "on_the_fly"
        else None,
    )
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # Loss
    loss_fn = CascadedLoss(
        pos_weight_disorder=None,
        pos_weight_binding=None,
        pos_weight_linker=None,
        loss_type=config["training"].get("loss_type", "bce"),
        focal_gamma=config["training"].get("focal_gamma", 2.0),
        focal_alpha=config["training"].get("focal_alpha", 0.25),
        device=device,
        idr_weight_binding=config["training"].get("idr_weight_binding", 1.0),
        idr_weight_linker=config["training"].get("idr_weight_linker", 1.0),
        disorder_loss_type=config["training"].get("disorder_loss_type", None),
        binding_loss_type=config["training"].get("binding_loss_type", None),
        linker_loss_type=config["training"].get("linker_loss_type", None),
        gaussian_sigma=config["training"].get("gaussian_sigma", 0.0),
        linker_gaussian_sigma=config["training"].get("linker_gaussian_sigma", 0.0),
    )

    # Optimizer
    opt_cfg = config["training"]["optimizer"]
    base_lr = opt_cfg["lr"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    num_epochs = config["training"]["num_epochs"]
    sched_cfg = config["training"].get("scheduler", {})
    eta_min = sched_cfg.get("eta_min", 1.0e-6)
    t_max = sched_cfg.get("T_max", num_epochs)
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    min_lr_ratio = eta_min / base_lr

    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, t_max - warmup_epochs))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    logger.info("=== Initializing trainer ===")
    trainer = MLPCascadeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=config["training"]["output_dir"],
        gradient_accumulation_steps=config["training"].get(
            "gradient_accumulation_steps", 1
        ),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        best_metric=config["training"].get("best_metric", "val_loss"),
        best_metric_mode=config["training"].get("best_metric_mode", "min"),
        scheduler=scheduler,
        composite_f1_weights=config["training"].get("composite_f1_weights", None),
        composite_auc_weights=config["training"].get("composite_auc_weights", None),
        model_config=config["model"],
        training_config=config,
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("=== Starting training ===")
    trainer.train(
        num_epochs=num_epochs,
        eval_every=config["training"].get("eval_every", 1),
        save_every=config["training"].get("save_every", 1),
        early_stopping_patience=config["training"].get("early_stopping_patience", 5),
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
