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
from src.experiments.ablation_no_cascade.model import cascDP_Ablation1
from src.experiments.ablation_no_cascade.loss import AblationLoss
from src.experiments.ablation_no_cascade.trainer import AblationTrainer
from src.models.backbone import create_backbone
from src.data.dataset import (
    DisorderFunctionDataset,
    OnTheFlyDisorderFunctionDataset,
    collate_fn,
)
from src.training.loss import compute_class_weights

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

    # Optional class-weight computation
    if (
        config["training"].get("use_class_weights", False)
        and config["training"].get("loss_type", "bce") == "bce"
    ):
        logger.info("Computing class weights from training data…")
        _, binding_weight, linker_weight = compute_class_weights(train_ds)
    else:
        binding_weight = linker_weight = None

    num_workers = 0 if mode == "on_the_fly" else config["training"].get("num_workers", 0)
    pin_memory = num_workers > 0

    # DisProt oversampling
    oversample_factor = config["training"].get("disprot_oversample_factor", 1)
    if oversample_factor > 1:
        sample_weights = []
        for pid in train_ds.protein_ids:
            source = train_ds.protein_sources.get(pid, "DisProt")
            sample_weights.append(float(oversample_factor) if "PDB" not in source else 1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        n_disprot = sum(1 for w in sample_weights if w > 1.0)
        n_pdb = len(sample_weights) - n_disprot
        logger.info(
            f"DisProt oversampling: factor={oversample_factor}x  "
            f"({n_disprot} DisProt, {n_pdb} PDB_missing entries)"
        )
        train_shuffle = False
        train_sampler = sampler
    else:
        train_shuffle = True
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, binding_weight, linker_weight

def create_model(config: dict, device: str) -> cascDP_Ablation1:
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
        layers_to_transform=lora_cfg.get("layers_to_transform"),
        target_modules=lora_cfg.get("target_modules"),
    )

    phase1_ckpt_path = config["model"].get("phase1_checkpoint")
    phase2_context = config["model"].get(
        "phase2_context_type",
        config["model"].get("context_type", "bigru"),
    )
    model_kwargs = dict(
        device=device,
        context_type=phase2_context,
        binding_context_type=config["model"].get("binding_context_type"),
        linker_context_type=config["model"].get("linker_context_type"),
        use_binding_head=config["model"].get("use_binding_head", True),
        use_linker_head=config["model"].get("use_linker_head", True),
        freeze_backbone=config["model"].get("freeze_backbone", True),
        dropout=config["model"].get("dropout", 0.5),
        binding_combined=config["model"].get("binding_combined", False),
        binding_head_type=config["model"].get("binding_head_type", "cnn"),
    )

    if phase1_ckpt_path:
        # Load only backbone.* keys from the Phase 1 checkpoint; no Phase1 instantiation.
        model = cascDP_Ablation1.from_phase1_checkpoint(
            checkpoint_path=phase1_ckpt_path,
            backbone=backbone,
            **model_kwargs,
        )
        logger.info(f"Backbone weights loaded from Phase 1 checkpoint: {phase1_ckpt_path}")
    else:
        logger.warning("No phase1_checkpoint — ablation model starts with random backbone!")
        model = cascDP_Ablation1(backbone=backbone, **model_kwargs)

    return model

def main():
    parser = argparse.ArgumentParser(description="Train ablation_no_cascade (no disorder cascade)")
    parser.add_argument("--config", required=True, help="Path to ablation YAML config")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training_ablation.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    set_seed(42)
    config = load_config(args.config)
    logger.info(f"Config: {args.config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Model
    logger.info("=== Creating ablation model ===")
    model = create_model(config, device)

    # Dataloaders
    logger.info("=== Loading data ===")
    backbone_for_emb = (
        model.backbone_wrapper
        if config["data"].get("embedding_mode") == "on_the_fly"
        else None
    )
    train_loader, val_loader, binding_weight, linker_weight = create_dataloaders(
        config, backbone=backbone_for_emb
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    training = config["training"]
    global_loss = training.get("loss_type", "bce")
    loss_fn = AblationLoss(
        binding_loss_type=training.get("binding_loss_type", global_loss),
        linker_loss_type=training.get("linker_loss_type", global_loss),
        pos_weight_binding=training.get("pos_weight_binding", binding_weight),
        pos_weight_linker=training.get("pos_weight_linker", linker_weight),
        idr_weight_binding=training.get("idr_weight_binding", 1.0),
        idr_weight_linker=training.get("idr_weight_linker", 1.0),
        linker_gaussian_sigma=training.get("linker_gaussian_sigma", 0.0),
        focal_gamma=training.get("focal_gamma", 2.0),
        focal_alpha=training.get("focal_alpha", 0.25),
        focal_positives_only=training.get("focal_positives_only", False),
        device=device,
    )

    opt_cfg = training["optimizer"]
    base_lr = opt_cfg["lr"]
    weight_decay = opt_cfg.get("weight_decay", 0.01)
    decouple_weight_decay = opt_cfg.get("decouple_weight_decay", False)

    if decouple_weight_decay:
        no_decay_names = {"bias"}
        no_decay_types = (torch.nn.LayerNorm,)
        decay_params, no_decay_params = [], []
        for module in model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param_name in no_decay_names or isinstance(module, no_decay_types):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        optimizer_params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        logger.info(
            f"Optimizer: decoupled weight decay "
            f"({len(decay_params)} decay, {len(no_decay_params)} no-decay, wd={weight_decay})"
        )
    else:
        optimizer_params = [
            {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay},
        ]
        logger.info(f"Optimizer: uniform weight decay (wd={weight_decay})")

    if opt_cfg["type"] == "AdamW":
        optimizer = torch.optim.AdamW(optimizer_params, lr=base_lr)
    else:
        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)

    num_epochs = training["num_epochs"]
    warmup_epochs = training.get("warmup_epochs", 0)
    scheduler_cfg = training.get("scheduler", {})
    eta_min = scheduler_cfg.get("eta_min", 1.0e-6)
    t_max = scheduler_cfg.get("T_max", num_epochs)
    min_lr_ratio = eta_min / base_lr

    if warmup_epochs > 0:
        logger.info(f"LR schedule: linear warmup {warmup_epochs}ep -> cosine to {eta_min:.1e} over {t_max - warmup_epochs}ep")
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, t_max - warmup_epochs))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"LR schedule: cosine over {t_max}ep to {eta_min:.1e}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    trainer = AblationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=training["output_dir"],
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        best_metric=training.get("best_metric", "binding_auc"),
        best_metric_mode=training.get("best_metric_mode", "max"),
        scheduler=scheduler,
        composite_auc_weights=training.get("composite_auc_weights"),
        model_config=config["model"],
        training_config=config,
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("=== Starting ablation training ===")
    trainer.train(
        num_epochs=num_epochs,
        eval_every=training.get("eval_every", 1),
        save_every=training.get("save_every", 1),
        early_stopping_patience=training.get("early_stopping_patience", 5),
    )
    logger.info("Done.")

if __name__ == "__main__":
    main()
