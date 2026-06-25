import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import json
import logging
import numpy as np
from tqdm import tqdm
from .loss import CascadedLoss
from ..evaluation.metrics import MetricsCalculator


logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model,  # cascDP_Phase1 or cascDP_Phase2
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: CascadedLoss,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        best_metric: str = 'val_loss',
        best_metric_mode: str = 'min',
        scheduler = None,
        **kwargs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True) 
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.model_config = kwargs.get('model_config', None)
        self.training_config = kwargs.get('training_config', None)
        
        from ..models.cascDP_phase1 import cascDP_Phase1
        from ..models.cascDP_phase2 import cascDP_Phase2
        
        if isinstance(model, cascDP_Phase1):
            logger.info("=== Training Phase 1 Model (disorder only) ===\n")
            self.is_phase1 = True
        elif isinstance(model, cascDP_Phase2):
            logger.info("=== Training Phase 2 Model (function heads) ===\n")
            self.is_phase1 = False
        else:
            raise TypeError(f"Model must be cascDP_Phase1 or cascDP_Phase2, got {type(model)}")
        
        if not self.is_phase1:
            self._param_groups_for_clip = []
            binding_params = [p for n, p in model.named_parameters()
                              if p.requires_grad and n.startswith('binding')]
            linker_params  = [p for n, p in model.named_parameters()
                              if p.requires_grad and n.startswith('linker')]
            other_params   = [p for n, p in model.named_parameters()
                              if p.requires_grad
                              and not n.startswith('binding')
                              and not n.startswith('linker')]
            if binding_params:
                self._param_groups_for_clip.append(binding_params)
            if linker_params:
                self._param_groups_for_clip.append(linker_params)
            if other_params:
                self._param_groups_for_clip.append(other_params)
            logger.info(f"Per-head gradient clipping: {len(self._param_groups_for_clip)} groups "
                        f"(binding={len(binding_params)}, linker={len(linker_params)}, other={len(other_params)})")
        else:
            self._param_groups_for_clip = None
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Best model tracking
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode  # 'min' or 'max'
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric_value = float('inf') if best_metric_mode == 'min' else float('-inf')
        self.best_threshold = 0.5          # disorder
        self.best_binding_threshold = 0.5   # binding
        self.best_linker_threshold = 0.5    # linker
        self.history = {
            'train_loss': [],
            'train_disorder_loss': [],
            'train_binding_loss': [],
            'train_linker_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        self.scheduler = kwargs.get('scheduler', None)
        
        self.composite_f1_weights = kwargs.get('composite_f1_weights', None)
        if self.composite_f1_weights:
            logger.info(f"Tracking composite F1 score with weights: {self.composite_f1_weights}")

        self.composite_auc_weights = kwargs.get('composite_auc_weights', None)
        if self.composite_auc_weights:
            logger.info(f"Tracking composite AUC score with weights: {self.composite_auc_weights}")

        if self.scheduler is not None:
            logger.info(f"Scheduler: {type(self.scheduler).__name__}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_disorder_loss = 0.0
        total_binding_loss = 0.0
        total_linker_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            disorder_labels = batch['disorder_labels'].to(self.device)
            function_labels = batch['function_labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            if 'loss_weight' in batch:
                 w = batch['loss_weight'].to(self.device) # [B]
                 mask = mask * w.unsqueeze(-1) # Broadcast to [B, L]
                 
            binding_mask = batch['binding_mask'].to(self.device) if 'binding_mask' in batch else None
            linker_mask = batch['linker_mask'].to(self.device) if 'linker_mask' in batch else None
            
            if 'sequences' in batch:
                inputs = batch['sequences']
            else:
                inputs = batch['embeddings'].to(self.device)
            
            # Forward pass
            if 'sequences' in batch:
                outputs = self.model(embeddings=None, sequences=inputs)
            else:
                outputs = self.model(inputs)
            
            # Unpack outputs - Phase 1: tensor, Phase 2: (disorder, binding, linker)
            if not isinstance(outputs, tuple):
                disorder_logits = outputs
                binding_logits = None
                linker_logits = None
            else:
                disorder_logits, binding_logits, linker_logits = outputs
            
            # Extract binding and linker labels from function_labels
            binding_combined = getattr(self.model, 'binding_combined', False)
            if binding_combined:
                binding_labels = function_labels[:, :, 4]    # (B, L) - combined
            else:
                binding_labels = function_labels[:, :, :4]   # (B, L, 4) - multi-label
            # Linker: Flexible linker at index 5
            linker_labels = function_labels[:, :, 5]  # (B, L)

            # Compute loss with separate masks
            loss, loss_dict = self.loss_fn(
                disorder_logits,
                binding_logits,
                linker_logits,
                disorder_labels,
                binding_labels,
                linker_labels,
                mask,
                binding_mask,
                linker_mask
            )
            
            # Check for NaN/Inf losses
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at batch {batch_idx}: {loss.item()}")
                self.optimizer.zero_grad()  # Clear any accumulated gradients
                continue  # Skip this batch
            
            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping per-head for Phase 2
                if self._param_groups_for_clip:
                    grad_norm = max(
                        torch.nn.utils.clip_grad_norm_(group, self.max_grad_norm)
                        for group in self._param_groups_for_clip
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                if self.global_step % 100 == 0:
                    logger.debug(f"Step {self.global_step} - Grad norm: {grad_norm:.4f}")
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            total_loss += loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
            total_disorder_loss += loss_dict['disorder'].item() if torch.is_tensor(loss_dict['disorder']) else loss_dict['disorder']
            total_binding_loss += loss_dict['binding'].item() if torch.is_tensor(loss_dict['binding']) else loss_dict['binding']
            total_linker_loss += loss_dict['linker'].item() if torch.is_tensor(loss_dict['linker']) else loss_dict['linker']
            num_batches += 1
            
            postfix_dict = {
                'loss': f"{loss_dict['total']:.4f}",
                'dis': f"{loss_dict['disorder']:.4f}",
                'bind': f"{loss_dict['binding']:.4f}",
                'link': f"{loss_dict['linker']:.4f}"
            }
            progress_bar.set_postfix(postfix_dict)
        
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            remaining = (batch_idx + 1) % self.gradient_accumulation_steps
            logger.debug(f"Applying remaining gradients from last {remaining} batch(es)")
            
            if self._param_groups_for_clip:
                for group in self._param_groups_for_clip:
                    torch.nn.utils.clip_grad_norm_(group, self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        # Average losses
        avg_losses = {
            'total': total_loss / num_batches,
            'disorder': total_disorder_loss / num_batches,
            'binding': total_binding_loss / num_batches,
            'linker': total_linker_loss / num_batches
        }
        
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.debug(f"Epoch {self.current_epoch} end - Learning rate: {current_lr:.2e}")
        
        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics_calc = MetricsCalculator()
        all_disorder_logits = []
        all_disorder_labels = []
        total_val_loss = 0.0
        total_disorder_loss = 0.0
        total_binding_loss = 0.0
        total_linker_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            disorder_labels = batch['disorder_labels'].to(self.device)
            function_labels = batch['function_labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Extract separate masks
            binding_mask = batch['binding_mask'].to(self.device) if 'binding_mask' in batch else None
            linker_mask = batch['linker_mask'].to(self.device) if 'linker_mask' in batch else None

            # Forward pass
            if 'sequences' in batch:
                outputs = self.model(embeddings=None, sequences=batch['sequences'])
            else:
                embeddings = batch['embeddings'].to(self.device)
                outputs = self.model(embeddings)
            
            # Unpack outputs - Phase 1: tensor, Phase 2: (disorder, binding, linker)
            if not isinstance(outputs, tuple):
                disorder_logits = outputs
                binding_logits = None
                linker_logits = None
            else:
                disorder_logits, binding_logits, linker_logits = outputs
            
            # Extract labels - branch on head mode
            binding_combined = getattr(self.model, 'binding_combined', False)
            if binding_combined:
                binding_labels = function_labels[:, :, 4]    # (B, L) - combined
            else:
                binding_labels = function_labels[:, :, :4]   # (B, L, 4) - individual types
            linker_labels = function_labels[:, :, 5]  # (B, L) - binary

            # Compute combined binding label for metrics (OR of all 4 types)
            binding_labels_combined = function_labels[:, :, 4]  # (B, L)
            
            # Compute loss with separate masks
            loss, loss_dict = self.loss_fn(
                disorder_logits,
                binding_logits,
                linker_logits,
                disorder_labels,
                binding_labels,
                linker_labels,
                mask,
                binding_mask,
                linker_mask
            )
            
            total_val_loss += loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
            total_disorder_loss += loss_dict['disorder'].item() if torch.is_tensor(loss_dict['disorder']) else loss_dict['disorder']
            total_binding_loss += loss_dict['binding'].item() if torch.is_tensor(loss_dict['binding']) else loss_dict['binding']
            total_linker_loss += loss_dict['linker'].item() if torch.is_tensor(loss_dict['linker']) else loss_dict['linker']
            num_batches += 1
            
            # Compute combined binding probability for metrics (OR of all 4 types)
            if self.is_phase1 or binding_logits is None:
                combined_binding_logits = None
                binding_logits_for_metrics = None
            else:
                if binding_combined:
                    # single combined head: logits already represent combined binding
                    combined_binding_logits = binding_logits  # (B, L, 1)
                    binding_logits_for_metrics = None  # no per-type
                else:
                    # multi-label: noisy-OR aggregation to combined
                    binding_probs_individual = torch.sigmoid(binding_logits)  # (B, L, 4)
                    combined_binding_probs = 1 - torch.prod(1 - binding_probs_individual, dim=-1, keepdim=True)
                    combined_binding_logits = torch.logit(combined_binding_probs.clamp(1e-7, 1 - 1e-7))
                    binding_logits_for_metrics = binding_logits
            
            # Handle CRF outputs
            if disorder_logits.dim() == 3 and disorder_logits.shape[-1] == 2:
                disorder_logits_for_metrics = (disorder_logits[:, :, 1:2] - disorder_logits[:, :, 0:1])  # Keep shape (B, L, 1)
            else:
                # Standard case: (B, L) or (B, L, 1)
                disorder_logits_for_metrics = disorder_logits
            
            # Handle CRF output for Linker
            if not self.is_phase1 and linker_logits is not None:
                if linker_logits.dim() == 3 and linker_logits.shape[-1] == 2:
                    linker_logits_for_metrics = (linker_logits[:, :, 1:2] - linker_logits[:, :, 0:1])
                else:
                     linker_logits_for_metrics = linker_logits
            else:
                 linker_logits_for_metrics = None
            
            # Update metrics with combined binding
            metrics_calc.update(
                disorder_logits_for_metrics,
                combined_binding_logits if not self.is_phase1 else None,
                linker_logits_for_metrics if not self.is_phase1 else None,
                disorder_labels,
                binding_labels_combined,  # Combined labels
                linker_labels,
                disorder_mask=mask,
                binding_mask=binding_mask if not self.is_phase1 else None,
                linker_mask=linker_mask if not self.is_phase1 else None,
                binding_logits_indiv=binding_logits_for_metrics if not self.is_phase1 else None,
                binding_labels_indiv=binding_labels if not self.is_phase1 else None
            )
            
            # Flatten for storage
            disorder_logits_flat = disorder_logits_for_metrics.squeeze(-1)[mask == 1]
            disorder_labels_flat = disorder_labels[mask == 1]
            all_disorder_logits.append(disorder_logits_flat)
            all_disorder_labels.append(disorder_labels_flat)

        metrics = {}
        if num_batches > 0:
            metrics['val_loss'] = total_val_loss / num_batches
            metrics['val_disorder_loss'] = total_disorder_loss / num_batches
            metrics['val_binding_loss'] = total_binding_loss / num_batches
            metrics['val_linker_loss'] = total_linker_loss / num_batches

        if len(all_disorder_logits) > 0:
            optimal_metrics = metrics_calc.compute_metrics()
            for k, v in optimal_metrics.items():
                metrics[k] = v
            
        return metrics
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        backbone_wrapper = self.model.phase1.backbone_wrapper if hasattr(self.model, 'phase1') else self.model.backbone_wrapper
        final_model_config = self.model_config.copy() if self.model_config else {}

        if 'backbone_type' not in final_model_config:
            final_model_config['backbone_type'] = backbone_wrapper.model_name.split('_')[0] if hasattr(backbone_wrapper, 'model_name') else 'esmc'
            
        final_model_config['architecture_version'] = '2.1'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_metric_value': self.best_metric_value,
            'best_threshold': getattr(self, 'best_threshold', 0.5),
            'best_binding_threshold': getattr(self, 'best_binding_threshold', 0.5),
            'best_linker_threshold': getattr(self, 'best_linker_threshold', 0.5),
            'history': self.history,
            'model_config': final_model_config,
            'training_config': self.training_config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        elif filename is not None:
            checkpoint_path = self.output_dir / filename
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric_value = checkpoint.get('best_metric_value', checkpoint.get('best_val_loss', float('inf')))
        self.best_threshold = checkpoint.get('best_threshold', 0.5)
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
        logger.info(f"Training will continue from epoch {self.current_epoch + 1}")
    
    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
        early_stopping_patience: int = 5
    ):
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_disorder_loss'].append(train_losses['disorder'])
            self.history['train_binding_loss'].append(train_losses['binding'])
            self.history['train_linker_loss'].append(train_losses['linker'])
    
            current_lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = (f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f} "
                       f"(Disorder: {train_losses['disorder']:.4f}, "
                       f"Binding: {train_losses['binding']:.4f}, "
                       f"Linker: {train_losses['linker']:.4f}")
            
            log_msg += f") LR: {current_lr:.2e}"
            logger.info(log_msg)
            
            # Validate
            if (epoch + 1) % eval_every == 0:
                val_results = self.validate()
                self.history['val_loss'].append(val_results['val_loss'])
                self.history['val_metrics'].append(val_results)
                
                logger.info(f"\nValidation Results:")
                logger.info(f"  Loss: {val_results['val_loss']:.4f} "
                           f"(Disorder: {val_results.get('val_disorder_loss', 0):.4f}, "
                           f"Binding: {val_results.get('val_binding_loss', 0):.4f}, "
                           f"Linker: {val_results.get('val_linker_loss', 0):.4f})")
                
                metrics_calc = MetricsCalculator()
                logger.info(f"\n{metrics_calc.format_metrics(val_results)}")
                
                # Check for improvement based on configured metric
                current_metric = val_results.get(self.best_metric, None)

                # Compute composite metrics on-the-fly if requested
                if current_metric is None and self.best_metric == 'val_composite_auc' and self.composite_auc_weights:
                    current_metric = sum(
                        val_results.get(k, 0.0) * w
                        for k, w in self.composite_auc_weights.items()
                    )
                    val_results['val_composite_auc'] = current_metric
                elif current_metric is None and self.best_metric == 'val_composite_f1' and self.composite_f1_weights:
                    current_metric = sum(
                        val_results.get(k, 0.0) * w
                        for k, w in self.composite_f1_weights.items()
                    )
                    val_results['val_composite_f1'] = current_metric

                if current_metric is None:
                    current_metric = val_results['val_loss']

                is_better = False
                
                if self.best_metric_mode == 'min':
                    is_better = current_metric < self.best_metric_value
                else:  # 'max'
                    is_better = current_metric > self.best_metric_value
                
                if is_better:
                    logger.info(f"New best {self.best_metric}: {current_metric:.4f} "
                               f"(previous: {self.best_metric_value:.4f})")
                    self.best_metric_value = current_metric
                    # Update best thresholds if available
                    if 'disorder_optimal_threshold' in val_results:
                        self.best_threshold = val_results['disorder_optimal_threshold']
                    if 'binding_optimal_threshold' in val_results:
                        self.best_binding_threshold = val_results['binding_optimal_threshold']
                    if 'linker_optimal_threshold' in val_results:
                        self.best_linker_threshold = val_results['linker_optimal_threshold']
                    self.save_checkpoint(is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.warning(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            if self.scheduler is not None:
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    logger.info(f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")
            
            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                # Convert to serializable format (handle numpy types)
                def convert_to_serializable(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                serializable_history = {
                    'train_loss': convert_to_serializable(self.history['train_loss']),
                    'train_disorder_loss': convert_to_serializable(self.history['train_disorder_loss']),
                    'train_binding_loss': convert_to_serializable(self.history['train_binding_loss']),
                    'train_linker_loss': convert_to_serializable(self.history['train_linker_loss']),
                    'val_loss': convert_to_serializable(self.history['val_loss']),
                    'val_metrics': convert_to_serializable(self.history['val_metrics'])
                }
                json.dump(serializable_history, f, indent=2)
        
        logger.info(f"\nTraining complete! Best {self.best_metric}: {self.best_metric_value:.4f}")
        if hasattr(self, 'best_threshold'):
            logger.info(f"Best threshold: {self.best_threshold:.4f}")