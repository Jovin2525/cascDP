import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from ...training.loss import Phase2Loss

class AblationLoss(nn.Module):
    def __init__(
        self,
        binding_loss_type: str = "bce",
        linker_loss_type: str = "bce",
        pos_weight_binding: Optional[float] = None,
        pos_weight_linker: Optional[float] = None,
        idr_weight_binding: float = 1.0,
        idr_weight_linker: float = 1.0,
        linker_gaussian_sigma: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_positives_only: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        self._loss = Phase2Loss(
            binding_loss_type=binding_loss_type,
            linker_loss_type=linker_loss_type,
            pos_weight_binding=pos_weight_binding,
            pos_weight_linker=pos_weight_linker,
            idr_weight_binding=idr_weight_binding,
            idr_weight_linker=idr_weight_linker,
            linker_gaussian_sigma=linker_gaussian_sigma,
            device=device,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            focal_positives_only=focal_positives_only,
        )

    def forward(
        self,
        binding_logits: Optional[torch.Tensor],
        linker_logits: Optional[torch.Tensor],
        binding_labels: torch.Tensor,
        linker_labels: torch.Tensor,
        mask: torch.Tensor,
        disorder_labels: Optional[torch.Tensor] = None,
        binding_mask: Optional[torch.Tensor] = None,
        linker_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        total_loss, loss_dict = self._loss(
            binding_logits=binding_logits,
            linker_logits=linker_logits,
            binding_labels=binding_labels,
            linker_labels=linker_labels,
            mask=mask,
            disorder_labels=disorder_labels,
            binding_mask=binding_mask,
            linker_mask=linker_mask,
        )

        loss_dict.setdefault("disorder", torch.tensor(0.0, device=self.device))
        return total_loss, loss_dict