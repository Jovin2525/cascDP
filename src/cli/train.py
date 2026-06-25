import os
import argparse
import math
import random
import yaml
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
from src.models.backbone import create_backbone
from src.models import cascDP_Phase1, cascDP_Phase1Recycle, cascDP_Phase2
from src.data.dataset import DisorderFunctionDataset, OnTheFlyDisorderFunctionDataset, collate_fn
from src.training.loss import CascadedLoss, compute_class_weights
from src.training.trainer import Trainer

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dataloaders(config: dict, backbone=None):
    logger = logging.getLogger(__name__)
    
    embedding_mode = config['data'].get('embedding_mode', 'precomputed')
    logger.info(f"Using embedding mode: {embedding_mode}")
    
    # Get dataset files from config
    train_file = config['data']['train_disorder_file']
    val_file = config['data']['val_disorder_file']
    
    logger.info(f"Train data: {train_file}")
    logger.info(f"Val data: {val_file}")
    
    if embedding_mode == 'precomputed':
        pdb_loss_weight = config['training'].get('pdb_loss_weight', 1.0)
        
        train_dataset = DisorderFunctionDataset(
            embedding_dir=config['data']['train_embedding_dir'],
            disorder_file=train_file,
            pdb_loss_weight=pdb_loss_weight
        )
        
        # Validation
        val_dataset = DisorderFunctionDataset(
            embedding_dir=config['data']['val_embedding_dir'],
            disorder_file=val_file,
            pdb_loss_weight=1.0 # Unbiased
        )
    
    elif embedding_mode == 'on_the_fly':
        # Generate embeddings on-the-fly
        if backbone is None:
            raise ValueError("on_the_fly mode requires backbone model to be provided")

        pdb_loss_weight = config['training'].get('pdb_loss_weight', 1.0)
        
        train_dataset = OnTheFlyDisorderFunctionDataset(
            disorder_file=train_file,
            embedding_model=backbone,
            device=backbone.device,
            pdb_loss_weight=pdb_loss_weight
        )
        
        val_dataset = OnTheFlyDisorderFunctionDataset(
            disorder_file=val_file,
            embedding_model=backbone,
            device=backbone.device,
            pdb_loss_weight=1.0
        )
    
    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}. Use 'precomputed' or 'on_the_fly'")
    
    if config['training'].get('use_class_weights', False) and config['training'].get('loss_type', 'bce') == 'bce':
        logger.info("Computing class weights from training data (for BCE loss)...")
        disorder_weight, binding_weight, linker_weight = compute_class_weights(train_dataset)
    else:
        disorder_weight = None
        binding_weight = None
        linker_weight = None
        _global_loss = config['training'].get('loss_type', 'bce')
        _disorder_loss = config['training'].get('disorder_loss_type', _global_loss)
        _linker_loss   = config['training'].get('linker_loss_type',   _global_loss)
        _binding_loss  = config['training'].get('binding_loss_type',  _global_loss)
        _disorder_crf  = config['model'].get('use_crf', False)
        _linker_crf    = config['model'].get('use_crf_linker', False)
        # Focal is active for a task only if that task's loss_type is focal and it does not use CRF
        _focal_tasks = []
        if _disorder_loss == 'focal' and not _disorder_crf:
            _focal_tasks.append('disorder')
        if _binding_loss == 'focal':
            _focal_tasks.append('binding')
        if _linker_loss == 'focal' and not _linker_crf:
            _focal_tasks.append('linker')
        _crf_tasks = [t for t, flag in [('disorder', _disorder_crf), ('linker', _linker_crf)] if flag]
        if _focal_tasks:
            logger.info(
                f"Using Focal Loss (alpha={config['training'].get('focal_alpha', 0.25)}) "
                f"for tasks: {', '.join(_focal_tasks)}"
                + (f" | CRF active for: {', '.join(_crf_tasks)}" if _crf_tasks else "")
            )
        elif _crf_tasks:
            logger.info(f"CRF active for: {', '.join(_crf_tasks)} (focal loss not used for any task)")
    
    embedding_mode = config['data'].get('embedding_mode', 'precomputed')
    effective_num_workers = 0 if embedding_mode == 'on_the_fly' else config['training'].get('num_workers', 0)
    use_pin_memory = effective_num_workers > 0  # pin_memory only helps when workers feed a GPU

    # Build per-sample weights for DisProt oversampling
    oversample_factor = config['training'].get('disprot_oversample_factor', 1)
    if oversample_factor > 1:
        sample_weights = []
        for pid in train_dataset.protein_ids:
            source = train_dataset.protein_sources.get(pid, 'DisProt')
            sample_weights.append(float(oversample_factor) if 'PDB' not in source else 1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        n_disprot = sum(1 for w in sample_weights if w > 1.0)
        n_pdb     = len(sample_weights) - n_disprot
        logger.info(
            f"DisProt oversampling: factor={oversample_factor}x  "
            f"({n_disprot} DisProt, {n_pdb} PDB_missing entries)"
        )
        train_shuffle  = False
        train_sampler  = sampler
    else:
        train_shuffle  = True
        train_sampler  = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=effective_num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, disorder_weight, binding_weight, linker_weight


def create_model(config: dict, device: str):
    logger = logging.getLogger(__name__)
    
    # Get training phase from config
    training_phase = config['training'].get('phase', 1)
    
    # Extract LoRA config
    lora_config = config['model'].get('lora', {})

    backbone = create_backbone(
        backbone_type=config['model']['backbone_type'],
        model_name=config['model']['backbone_name'],
        device=device,
        use_lora=config['model'].get('use_lora', False),
        lora_r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        layers_to_transform=lora_config.get('layers_to_transform', None),
        target_modules=lora_config.get('target_modules', None)
    )
    
    if training_phase == 1:
        # Phase 1: Create disorder-only model
        logger.info("Creating Phase 1 model (backbone + disorder head)")
        model = cascDP_Phase1(
            backbone=backbone,
            device=device,
            context_type=config['model'].get('context_type', 'bigru'),
            dropout=config['model'].get('dropout', 0.5),
            use_crf=config['model'].get('use_crf', False),
            freeze_backbone=config['model'].get('freeze_backbone', False),
            disorder_prior=config['model'].get('disorder_prior', 0.1024),
            fusion_type=config['model'].get('fusion_type', 'sum'),
        )
    
    elif training_phase == 2:
        # Phase 2: Create Phase 1 model first, then attach function heads
        logger.info("Creating Phase 2 model (function heads on top of Phase 1)")
        
        # Determine Phase 1 configuration from checkpoint if possible
        phase1_checkpoint = config['model'].get('phase1_checkpoint', None)
        phase1_use_crf = config['model'].get('use_crf', False) # Default to config, override if checkpoint found
        
        checkpoint = None
        if phase1_checkpoint:
             logger.info(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
             checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)
             
             # Detect CRF from checkpoint keys if not explicitly in config
             checkpoint_keys = checkpoint['model_state_dict'].keys()
             detected_crf = any('crf.transitions' in key for key in checkpoint_keys)
             if detected_crf:
                 logger.info("Detected CRF in Phase 1 checkpoint - overriding config to use_crf=True")
                 phase1_use_crf = True
        
        # Determine Phase 1 context type
        # Priority: Checkpoint Config > Explicit Config > Default
        phase1_context_type = config['model'].get('phase1_context_type', config['model'].get('context_type', 'bigru'))
        
        if checkpoint and 'model_config' in checkpoint:
            ckpt_ctx = checkpoint['model_config'].get('phase1_context_type', checkpoint['model_config'].get('context_type'))
            if ckpt_ctx:
                phase1_context_type = ckpt_ctx
                logger.info(f"Using Phase 1 context_type from checkpoint: {phase1_context_type}")
        else:
            logger.info(f"Using Phase 1 context_type from config: {phase1_context_type}")

        # Detect if the phase 1 checkpoint is a recycled model
        is_recycled_phase1 = (checkpoint is not None and
            any('recycle_proj' in k for k in checkpoint['model_state_dict'].keys()))
        if is_recycled_phase1:
            num_recycles = checkpoint.get('model_config', {}).get('num_recycles', 2)
            logger.info(f"Detected recycled Phase 1 checkpoint (num_recycles={num_recycles})")
            phase1_model = cascDP_Phase1Recycle(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                dropout=config['model'].get('dropout', 0.5),
                use_crf=phase1_use_crf,
                num_recycles=num_recycles,
            )
        else:
            phase1_model = cascDP_Phase1(
                backbone=backbone,
                device=device,
                context_type=phase1_context_type,
                dropout=config['model'].get('dropout', 0.5),
                use_crf=phase1_use_crf,
                fusion_type=config['model'].get('fusion_type', 'sum'),
            )

        # Load Phase 1 checkpoint weights
        if checkpoint:
            phase1_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            logger.info("Phase 1 checkpoint loaded successfully")
        else:
            logger.warning("No phase1_checkpoint specified - starting Phase 2 from scratch (not recommended)")
        
        # Create Phase 2 model (attaches function heads)
        model = cascDP_Phase2(
            phase1_model=phase1_model,
            device=device,
            context_type=config['model'].get('phase2_context_type', 'bigru'),
            dropout=config['model'].get('dropout', 0.2),
            use_binding_head=config['model'].get('use_binding_head', True),
            use_linker_head=config['model'].get('use_linker_head', True),
            use_crf_linker=config['model'].get('use_crf_linker', False),
            binding_combined=config['model'].get('binding_combined', False),
            binding_head_type=config['model'].get('binding_head_type', 'cnn'),
            binding_priors=config['model'].get('binding_priors'),
            binding_combined_prior=config['model'].get('binding_combined_prior', 0.1299),
            linker_prior=config['model'].get('linker_prior', 0.0218),
        )
    
    else:
        raise ValueError(f"Unknown training_phase: {training_phase}. Use 1 or 2.")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train cascDP model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--phase', type=int, default=None,
                       help='Training phase: 1 (backbone+disorder) or 2 (frozen backbone+disorder, train linker/binding). Overrides config.')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.log_level == 'DEBUG' else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)

    seed = 42
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override training phase from command line if specified
    if args.phase is not None:
        config['training']['phase'] = args.phase
        logger.info(f"Training phase overridden via --phase argument: {args.phase}")
    
    # Get training phase (default to 1 if not specified)
    training_phase = config['training'].get('phase', 1)
    logger.info(f"Training Phase: {training_phase}")
    if training_phase == 1:
        logger.info("Phase 1: Training backbone + disorder head")
    elif training_phase == 2:
        logger.info("Phase 2: Frozen backbone+disorder, training linker head (and binding if enabled)")
    
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("Using CUDA device selected by CUDA_VISIBLE_DEVICES")
    else:
        device = 'cpu'
        logger.info(f"CUDA not available, using CPU")
    
    logger.info(f"Device: {device}")
    
    # Create model
    logger.info("\nCreating Model...")
    model = create_model(config, device)
    
    # Get backbone reference for on_the_fly mode
    if hasattr(model, 'backbone_wrapper'):
        backbone_for_embedding = model.backbone_wrapper
    elif hasattr(model, 'phase1') and hasattr(model.phase1, 'backbone_wrapper'):
        backbone_for_embedding = model.phase1.backbone_wrapper
    else:
        backbone_for_embedding = None
    
    # Create dataloaders (pass backbone for on_the_fly mode if needed)
    logger.info("\nLoading Data...")
    train_loader, val_loader, disorder_weight, binding_weight, linker_weight = create_dataloaders(
        config, 
        backbone=backbone_for_embedding if config['data'].get('embedding_mode') == 'on_the_fly' else None
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Extract CRFs if available
    disorder_crf = None
    linker_crf = None
    
    if hasattr(model, 'crf'): # Phase 1
        disorder_crf = model.crf
    elif hasattr(model, 'phase1') and hasattr(model.phase1, 'crf'): # Phase 2
        disorder_crf = model.phase1.crf
        
    if hasattr(model, 'linker_crf'): # Phase 2
        linker_crf = model.linker_crf

    # Create loss function
    # Prefer config weights if specified, otherwise use computed/None
    pos_weight_disorder = config['training'].get('pos_weight_disorder', disorder_weight)
    pos_weight_binding = config['training'].get('pos_weight_binding', binding_weight)
    pos_weight_linker = config['training'].get('pos_weight_linker', linker_weight)

    boundary_aux_weight = config['training'].get('boundary_aux_weight', 0.0)
    boundary_radius     = config['training'].get('boundary_radius', 1)
    if boundary_aux_weight > 0.0:
        logger.info(
            f"Boundary auxiliary loss enabled: weight={boundary_aux_weight}, "
            f"radius={boundary_radius} residue(s) from D<->O transitions"
        )

    loss_fn = CascadedLoss(
        pos_weight_disorder=pos_weight_disorder,
        pos_weight_binding=pos_weight_binding,
        pos_weight_linker=pos_weight_linker,
        loss_type=config['training'].get('loss_type', 'bce'),
        focal_gamma=config['training'].get('focal_gamma', 2.0),
        focal_alpha=config['training'].get('focal_alpha', 0.25),
        device=device,
        idr_weight_binding=config['training'].get('idr_weight_binding', 1.0),
        idr_weight_linker=config['training'].get('idr_weight_linker', 1.0),
        disorder_crf=disorder_crf,
        linker_crf=linker_crf,
        # Granular loss type config
        disorder_loss_type=config['training'].get('disorder_loss_type', None),
        binding_loss_type=config['training'].get('binding_loss_type', None),
        linker_loss_type=config['training'].get('linker_loss_type', None),
        # Boundary auxiliary loss
        boundary_aux_weight=boundary_aux_weight,
        boundary_radius=boundary_radius,
        gaussian_sigma=config['training'].get('gaussian_sigma', 0.0),
        linker_gaussian_sigma=config['training'].get('linker_gaussian_sigma', 0.0),
        focal_positives_only=config['training'].get('focal_positives_only', False),
    )
    
    # Create optimizer
    optimizer_config = config['training']['optimizer']
    base_lr = optimizer_config['lr']
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    decouple_weight_decay = optimizer_config.get('decouple_weight_decay', False)

    if decouple_weight_decay:
        no_decay_names = {'bias'}
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
            {'params': decay_params,    'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        logger.info(f"Optimizer: decoupled weight decay "
                    f"({len(decay_params)} decay, {len(no_decay_params)} no-decay, wd={weight_decay})")
    else:
        # Single param group: weight_decay applied uniformly to all trainable params
        optimizer_params = [
            {'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': weight_decay},
        ]
        logger.info(f"Optimizer: uniform weight decay (wd={weight_decay})")

    if optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_params, lr=base_lr)
    elif optimizer_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Create learning rate scheduler
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    scheduler_cfg = config['training'].get('scheduler', {})
    eta_min = scheduler_cfg.get('eta_min', 1.0e-6)
    t_max = scheduler_cfg.get('T_max', num_epochs)
    min_lr_ratio = eta_min / base_lr

    if warmup_epochs > 0:
        logger.info(f"LR schedule: linear warmup {warmup_epochs}ep -> cosine to {eta_min:.1e} over {t_max - warmup_epochs}ep")
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            progress = float(current_epoch - warmup_epochs) / float(max(1, t_max - warmup_epochs))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_factor)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"LR schedule: cosine over {t_max}ep to {eta_min:.1e}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    
    # Create trainer
    logger.info("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=config['training']['output_dir'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        best_metric=config['training'].get('best_metric', 'val_loss'),
        best_metric_mode=config['training'].get('best_metric_mode', 'min'),
        scheduler=scheduler,
        composite_f1_weights=config['training'].get('composite_f1_weights', None),
        composite_auc_weights=config['training'].get('composite_auc_weights', None),
        model_config=config['model'], # Pass model config to be saved in checkpoint
        training_config=config,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("\nStarting Training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        eval_every=config['training'].get('eval_every', 1),
        save_every=config['training'].get('save_every', 1),
        early_stopping_patience=config['training'].get('early_stopping_patience', 5)
    )
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()
