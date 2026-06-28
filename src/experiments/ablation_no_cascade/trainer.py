import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ...evaluation.metrics import MetricsCalculator
from .loss import AblationLoss

logger = logging.getLogger(__name__)

class AblationTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: AblationLoss,
        device: str = "cuda",
        output_dir: str = "./checkpoints/ablation_no_cascade",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        best_metric: str = "binding_auc",
        best_metric_mode: str = "max",
        scheduler=None,
        composite_auc_weights: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
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
        self.model_config = model_config or {}
        self.training_config = training_config
        self.composite_auc_weights = composite_auc_weights

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("=== Training Ablation1 Model (no disorder cascade) ===")
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode
        self.best_metric_value = float("inf") if best_metric_mode == "min" else float("-inf")
        self.best_binding_threshold = 0.5
        self.best_linker_threshold = 0.5

        self.current_epoch = 0
        self.global_step = 0
        self.history: Dict = {"train_loss": [], "val_loss": [], "val_metrics": []}

        if scheduler is not None:
            logger.info(f"Scheduler: {type(scheduler).__name__}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = total_binding_loss = total_linker_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            disorder_labels = batch["disorder_labels"].to(self.device)
            function_labels = batch["function_labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            if "loss_weight" in batch:
                w = batch["loss_weight"].to(self.device)
                mask = mask * w.unsqueeze(-1)

            binding_mask = batch["binding_mask"].to(self.device) if "binding_mask" in batch else None
            linker_mask = batch["linker_mask"].to(self.device) if "linker_mask" in batch else None

            if "sequences" in batch:
                _, binding_logits, linker_logits = self.model(
                    embeddings=None, sequences=batch["sequences"]
                )
            else:
                _, binding_logits, linker_logits = self.model(
                    embeddings=batch["embeddings"].to(self.device)
                )

            binding_combined = getattr(self.model, 'binding_combined', False)
            if binding_combined:
                binding_labels = function_labels[:, :, 4]    # (B, L)
            else:
                binding_labels = function_labels[:, :, :4]   # (B, L, 4)
            linker_labels = function_labels[:, :, 5]     # (B, L)

            loss, loss_dict = self.loss_fn(
                binding_logits=binding_logits,
                linker_logits=linker_logits,
                binding_labels=binding_labels,
                linker_labels=linker_labels,
                mask=mask,
                disorder_labels=disorder_labels,   # labels only — IDR weighting
                binding_mask=binding_mask,
                linker_mask=linker_mask,
            )

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at batch {batch_idx} — skipping")
                self.optimizer.zero_grad()
                continue

            (loss / self.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss_dict["total"].item() if torch.is_tensor(loss_dict["total"]) else loss_dict["total"]
            total_binding_loss += loss_dict.get("binding", torch.tensor(0.0)).item()
            total_linker_loss += loss_dict.get("linker", torch.tensor(0.0)).item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "bind": f"{loss_dict.get('binding', 0):.4f}",
                "link": f"{loss_dict.get('linker', 0):.4f}",
            })

        # Flush remaining accumulated gradients
        if num_batches > 0 and (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        n = max(num_batches, 1)
        return {
            "total": total_loss / n,
            "disorder": 0.0,   # always zero — kept for log compatibility
            "binding": total_binding_loss / n,
            "linker": total_linker_loss / n,
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_val_loss = total_binding_loss = total_linker_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            disorder_labels = batch["disorder_labels"].to(self.device)
            function_labels = batch["function_labels"].to(self.device)
            mask = batch["mask"].to(self.device)
            binding_mask = batch["binding_mask"].to(self.device) if "binding_mask" in batch else None
            linker_mask = batch["linker_mask"].to(self.device) if "linker_mask" in batch else None

            if "sequences" in batch:
                _, binding_logits, linker_logits = self.model(
                    embeddings=None, sequences=batch["sequences"]
                )
            else:
                _, binding_logits, linker_logits = self.model(
                    embeddings=batch["embeddings"].to(self.device)
                )

            binding_combined = getattr(self.model, 'binding_combined', False)
            if binding_combined:
                binding_labels = function_labels[:, :, 4]
            else:
                binding_labels = function_labels[:, :, :4]
            linker_labels = function_labels[:, :, 5]
            binding_labels_combined = function_labels[:, :, 4]

            loss, loss_dict = self.loss_fn(
                binding_logits=binding_logits,
                linker_logits=linker_logits,
                binding_labels=binding_labels,
                linker_labels=linker_labels,
                mask=mask,
                disorder_labels=disorder_labels,
                binding_mask=binding_mask,
                linker_mask=linker_mask,
            )

            total_val_loss += loss_dict["total"].item() if torch.is_tensor(loss_dict["total"]) else loss_dict["total"]
            total_binding_loss += loss_dict.get("binding", torch.tensor(0.0)).item()
            total_linker_loss += loss_dict.get("linker", torch.tensor(0.0)).item()
            num_batches += 1

            combined_binding_logits = None
            binding_logits_indiv = None
            if binding_logits is not None:
                if binding_combined:
                    combined_binding_logits = binding_logits
                else:
                    binding_logits_indiv = binding_logits
                    binding_probs_indiv = torch.sigmoid(binding_logits)
                    combined_binding_probs = 1 - torch.prod(1 - binding_probs_indiv, dim=-1, keepdim=True)
                    combined_binding_logits = torch.logit(combined_binding_probs.clamp(1e-7, 1 - 1e-7))

            if linker_logits is not None and linker_logits.dim() == 3 and linker_logits.shape[-1] == 2:
                linker_logits_metrics = linker_logits[:, :, 1:2] - linker_logits[:, :, 0:1]
            else:
                linker_logits_metrics = linker_logits

            metrics_calc.update(
                disorder_logits=None,
                binding_logits=combined_binding_logits,
                linker_logits=linker_logits_metrics,
                disorder_labels=disorder_labels,
                binding_labels=binding_labels_combined,
                linker_labels=linker_labels,
                disorder_mask=None,
                binding_mask=binding_mask,
                linker_mask=linker_mask,
                binding_logits_indiv=binding_logits_indiv,
                binding_labels_indiv=None if binding_combined else binding_labels,
            )

        n = max(num_batches, 1)
        metrics: Dict = {
            "val_loss": total_val_loss / n,
            "val_disorder_loss": 0.0,
            "val_binding_loss": total_binding_loss / n,
            "val_linker_loss": total_linker_loss / n,
        }
        metrics.update(metrics_calc.compute_metrics())
        return metrics

    def save_checkpoint(self, filename: Optional[str] = None, is_best: bool = False):
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"

        cfg = self.model_config.copy()
        bw = self.model.backbone_wrapper
        cfg.setdefault(
            "backbone_type",
            bw.model_name.split("_")[0] if hasattr(bw, "model_name") else "esmc",
        )
        cfg["ablation"] = "no_disorder_cascade"
        cfg["architecture_version"] = "ablation_no_cascade"

        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_metric_value": self.best_metric_value,
            "best_binding_threshold": self.best_binding_threshold,
            "best_linker_threshold": self.best_linker_threshold,
            "history": self.history,
            "model_config": cfg,
            "training_config": self.training_config,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.output_dir / filename
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_metric_value = ckpt.get("best_metric_value", ckpt.get("best_val_loss", float("inf")))
        self.best_binding_threshold = ckpt.get("best_binding_threshold", 0.5)
        self.best_linker_threshold = ckpt.get("best_linker_threshold", 0.5)
        self.history = ckpt["history"]

        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
        early_stopping_patience: int = 5,
    ):
        logger.info(f"Starting ablation training: {num_epochs} epochs")
        logger.info(f"Device: {self.device} | Output: {self.output_dir}")
        logger.info(f"Best metric: {self.best_metric} ({self.best_metric_mode})")

        patience_counter = 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            self.history["train_loss"].append(train_losses)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} — Loss: {train_losses['total']:.4f} "
                f"(bind={train_losses['binding']:.4f}, link={train_losses['linker']:.4f}) "
                f"LR={current_lr:.2e}"
            )

            if (epoch + 1) % eval_every == 0:
                val_results = self.validate()
                self.history["val_loss"].append(val_results["val_loss"])
                self.history["val_metrics"].append(val_results)

                logger.info(
                    f"Val Loss: {val_results['val_loss']:.4f} "
                    f"(bind={val_results.get('val_binding_loss', 0):.4f}, "
                    f"link={val_results.get('val_linker_loss', 0):.4f})"
                )
                logger.info(MetricsCalculator().format_metrics(val_results))

                current_metric = val_results.get(self.best_metric, val_results["val_loss"])
                logger.info(
                    f"Metric '{self.best_metric}': {current_metric:.6f} "
                    f"(best: {self.best_metric_value:.6f})"
                )

                is_better = (
                    current_metric < self.best_metric_value
                    if self.best_metric_mode == "min"
                    else current_metric > self.best_metric_value
                )

                if is_better:
                    self.best_metric_value = current_metric
                    if "binding_optimal_threshold" in val_results:
                        self.best_binding_threshold = val_results["binding_optimal_threshold"]
                    if "linker_optimal_threshold" in val_results:
                        self.best_linker_threshold = val_results["linker_optimal_threshold"]
                    self.save_checkpoint(is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.warning(f"Early stopping after epoch {epoch + 1}")
                    break

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()

            if self.scheduler is not None:
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr != current_lr:
                    logger.info(f"LR: {current_lr:.2e} -> {new_lr:.2e}")

            self._save_history()

        logger.info(f"Training complete. Best {self.best_metric}: {self.best_metric_value:.4f}")

    def _save_history(self):
        def _s(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _s(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_s(i) for i in obj]
            return obj

        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(_s(self.history), f, indent=2)