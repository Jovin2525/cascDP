import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Standard BCE Calculation
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # Scalar alpha balancing
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype),
                              torch.tensor(1 - self.alpha, device=inputs.device, dtype=inputs.dtype))

        # Calculate Element-wise Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss

def get_loss_criterion(
    loss_type: str,
    pos_weight: float = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    device: str = 'cuda'
) -> nn.Module:

    if loss_type == 'focal':
        return FocalLoss(alpha=float(focal_alpha), gamma=focal_gamma)

    elif loss_type == 'bce':
        pw = None
        if pos_weight is not None and float(pos_weight) != 1.0:
            pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
        if pw is not None:
            return nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
        else:
            return nn.BCEWithLogitsLoss(reduction='none')
            
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

class Phase1Loss(nn.Module):
    def __init__(
        self,
        loss_type: str = 'bce',
        pos_weight: float = None,
        crf: nn.Module = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        gaussian_sigma: float = 0.0,
        device: str = 'cuda',
        boundary_aux_weight: float = 0.0,
        boundary_radius: int = 1,
    ):
        super().__init__()
        self.device = device
        self.crf = crf
        self.loss_type = loss_type
        self.boundary_aux_weight = boundary_aux_weight
        self.boundary_radius = boundary_radius

        self.criterion = get_loss_criterion(
            loss_type=loss_type,
            pos_weight=pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            device=device
        )

        # Gaussian spatial smoothing applied to labels before any loss function
        # When gaussian_sigma > 0, labels near D<->O boundaries are softened;
        # interior residues remain ~0 or ~1.
        if gaussian_sigma > 0:
            radius = max(1, int(math.ceil(3.0 * gaussian_sigma)))
            x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (x / gaussian_sigma) ** 2)
            k = k / k.sum()
            self.register_buffer('_gauss_kernel', k.view(1, 1, -1))
            self._gauss_pad = radius
        else:
            self._gauss_kernel = None

        if boundary_aux_weight > 0.0:
            self.boundary_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.boundary_criterion = None

    def _boundary_aux_loss(self, emissions, labels, mask):
        # Focal loss restricted to positions within `boundary_radius` of a D<->O transition

        # Detect transition sites i.e. positions where label changes
        transitions = (labels[:, 1:] != labels[:, :-1]).float()

        # Mark every position within boundary_radius of a transition
        boundary = torch.zeros_like(labels, dtype=torch.float)  # (B, L)
        for k in range(1, self.boundary_radius + 1):
            # Position to left of transition
            boundary[:, :labels.shape[1] - k] += transitions[:, k - 1:]
            # Position to right of transition
            boundary[:, k:] += transitions[:, :labels.shape[1] - k]
        boundary = boundary.clamp(0, 1)
        boundary_mask = mask * boundary
        if boundary_mask.sum() < 1:
            return torch.tensor(0.0, device=emissions.device)
        
        if emissions.dim() == 3 and emissions.shape[-1] == 2:
            logit = emissions[:, :, 1] - emissions[:, :, 0]  # (B, L)
        elif emissions.dim() == 3 and emissions.shape[-1] == 1:
            logit = emissions.squeeze(-1)
        else:
            logit = emissions  # already (B, L)

        raw = self.boundary_criterion(logit, labels.float())  # (B, L)
        return (raw * boundary_mask).sum() / boundary_mask.sum().clamp(min=1.0)

    def _smooth_labels(self, labels):
        # Apply Gaussian spatial smoothing to binary labels (B, L)
        if self._gauss_kernel is None:
            return labels
        smoothed = F.conv1d(
            labels.float().unsqueeze(1),
            self._gauss_kernel,
            padding=self._gauss_pad,
        ).squeeze(1).clamp(0.0, 1.0)
        return smoothed

    def forward(self, logits, labels, mask):
        smooth_labels = self._smooth_labels(labels)

        if self.crf is not None:
             mask_bool = (mask > 0).bool()
             sample_weights = mask.max(dim=-1).values  # (B,) — same w for all tokens in a sample
             log_likelihood = self.crf(logits, labels.long(), mask=mask_bool, reduction='none')  # (B,)
             nll_per_sample = -log_likelihood  # (B,)
             token_counts = mask_bool.float().sum(dim=-1)  # (B,)
             total_weighted_tokens = (token_counts * sample_weights).sum()
             nll = (nll_per_sample * sample_weights).sum() / torch.clamp(total_weighted_tokens, min=1.0)
             if self.boundary_aux_weight > 0.0:
                 nll = nll + self.boundary_aux_weight * self._boundary_aux_loss(logits, labels, mask_bool.float())
             return nll
             
        if logits.dim() == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        raw_loss = self.criterion(logits, smooth_labels)
        masked_loss = raw_loss * mask

        if self.loss_type == 'focal':
            num_positives = (labels * mask).sum()
            main_loss = masked_loss.sum() / torch.clamp(num_positives, min=1.0)
        else:
            # default: plain mean over valid tokens
            main_loss = masked_loss.sum() / torch.clamp(mask.sum(), min=1.0)

        if self.boundary_aux_weight > 0.0:
            main_loss = main_loss + self.boundary_aux_weight * self._boundary_aux_loss(logits, labels, mask)

        return main_loss

class Phase2Loss(nn.Module):
    def __init__(
        self,
        binding_loss_type: str = 'bce',
        linker_loss_type: str = 'bce',
        pos_weight_binding: float = None,
        pos_weight_linker: float = None,
        idr_weight_binding: float = 1.0,
        idr_weight_linker: float = 1.0,
        linker_crf: nn.Module = None,
        linker_gaussian_sigma: float = 0.0,
        device: str = 'cuda',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_positives_only: bool = False,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.idr_weight_binding = idr_weight_binding
        self.idr_weight_linker = idr_weight_linker
        self.linker_crf = linker_crf
        self.binding_loss_type = binding_loss_type
        self.linker_loss_type = linker_loss_type
        self.focal_positives_only = focal_positives_only

        # Gaussian label smoothing for linker (same approach as Phase 1 disorder).
        # Softens L<->non-L boundaries; skipped when CRF is active (requires hard labels).
        if linker_gaussian_sigma > 0 and linker_crf is None:
            radius = max(1, int(math.ceil(3.0 * linker_gaussian_sigma)))
            x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (x / linker_gaussian_sigma) ** 2)
            k = k / k.sum()
            self.register_buffer('_linker_gauss_kernel', k.view(1, 1, -1))
            self._linker_gauss_pad = radius
            logger.info(f"Phase2Loss: Gaussian label smoothing for linker (sigma={linker_gaussian_sigma})")
        else:
            self._linker_gauss_kernel = None

        # Binding Criterion
        self.binding_criterion = get_loss_criterion(
            loss_type=binding_loss_type,
            pos_weight=pos_weight_binding,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            device=device,
        )
        
        # Linker Criterion
        self.linker_criterion = get_loss_criterion(
            loss_type=linker_loss_type,
            pos_weight=pos_weight_linker,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            device=device,
        )

    def _smooth_linker_labels(self, labels):
        # Apply Gaussian spatial smoothing to binary linker labels (B, L)
        if self._linker_gauss_kernel is None:
            return labels
        smoothed = F.conv1d(
            labels.float().unsqueeze(1),
            self._linker_gauss_kernel,
            padding=self._linker_gauss_pad,
        ).squeeze(1).clamp(0.0, 1.0)
        return smoothed

    def _compute_loss(self, criterion, loss_type, logits, labels, mask, idr_weight=1.0, disorder_labels=None):
        if logits.dim() == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        # Effective Mask (Mask * IDR Weight)
        effective_mask = mask.float()
        if idr_weight != 1.0 and disorder_labels is not None:
             weight_map = 1.0 + (idr_weight - 1.0) * disorder_labels
             effective_mask = effective_mask * weight_map
             
        # Compute Loss (pixel-wise)
        raw_loss = criterion(logits, labels)
        
        # Reduction
        if raw_loss.dim() == 3 and effective_mask.dim() == 2:
            effective_mask = effective_mask.unsqueeze(-1)
            
        masked_loss = raw_loss * effective_mask
        
        if loss_type == 'focal':
            if self.focal_positives_only:
                eff_pos_mask = (effective_mask > 0).float()
                if labels.dim() > eff_pos_mask.dim():
                    eff_pos_mask = eff_pos_mask.unsqueeze(-1)
                num_positives = (labels * eff_pos_mask).sum()
            else:
                num_positives = (labels * (mask > 0).float().unsqueeze(-1) if labels.dim() > mask.dim() else mask).sum()
            return masked_loss.sum() / torch.clamp(num_positives, min=1.0)
        
        # Mean over weight mass
        return masked_loss.sum() / torch.clamp(effective_mask.sum(), min=1.0)

    def forward(self, binding_logits, linker_logits, binding_labels, linker_labels, 
                mask, disorder_labels=None, binding_mask=None, linker_mask=None):
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Binding
        if binding_logits is not None:
             b_mask = mask * binding_mask if binding_mask is not None else mask
             b_loss = self._compute_loss(
                 self.binding_criterion, 
                 self.binding_loss_type,
                 binding_logits, 
                 binding_labels, 
                 b_mask, 
                 idr_weight=self.idr_weight_binding,
                 disorder_labels=disorder_labels
             )
             total_loss += b_loss
             loss_dict['binding'] = b_loss.detach()
             
        # Linker
        if linker_logits is not None:
             l_mask = mask * linker_mask if linker_mask is not None else mask
             
             if self.linker_crf is not None:
                  mask_bool = l_mask.bool()
                  loss_dict['linker'] = -self.linker_crf(linker_logits, linker_labels.long(), mask=mask_bool, reduction='token_mean')
             else:
                  smooth_linker_labels = self._smooth_linker_labels(linker_labels)
                  loss_dict['linker'] = self._compute_loss(
                     self.linker_criterion,
                     self.linker_loss_type,
                     linker_logits,
                     smooth_linker_labels,
                     l_mask,
                     idr_weight=self.idr_weight_linker,
                     disorder_labels=disorder_labels
                  )
             total_loss += loss_dict['linker']
             
        loss_dict['total'] = total_loss.detach()
        return total_loss, loss_dict

class CascadedLoss(nn.Module):
    def __init__(
        self,
        pos_weight_disorder: float = None,
        pos_weight_binding: float = None,
        pos_weight_linker: float = None,
        loss_type: str = 'bce', # Default global type
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        device: str = 'cuda',
        idr_weight_binding: float = 1.0,
        idr_weight_linker: float = 1.0,
        disorder_crf: nn.Module = None,
        linker_crf: nn.Module = None,
        disorder_loss_type: str = None,
        binding_loss_type: str = None,
        linker_loss_type: str = None,
        boundary_aux_weight: float = 0.0,
        boundary_radius: int = 1,
        gaussian_sigma: float = 0.0,
        linker_gaussian_sigma: float = 0.0,
        focal_positives_only: bool = False,
    ):
        super().__init__()
        self.device = device
        
        # Determine specific loss types (fallback to global loss_type)
        self.disorder_loss_type = disorder_loss_type or loss_type
        self.binding_loss_type = binding_loss_type or loss_type
        self.linker_loss_type = linker_loss_type or loss_type

        self.phase1_loss = Phase1Loss(
            loss_type=self.disorder_loss_type,
            pos_weight=pos_weight_disorder,
            crf=disorder_crf,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            gaussian_sigma=gaussian_sigma,
            device=device,
            boundary_aux_weight=boundary_aux_weight,
            boundary_radius=boundary_radius,
        )
        
        self.phase2_loss = Phase2Loss(
            binding_loss_type=self.binding_loss_type,
            linker_loss_type=self.linker_loss_type,
            pos_weight_binding=pos_weight_binding,
            pos_weight_linker=pos_weight_linker,
            idr_weight_binding=idr_weight_binding,
            idr_weight_linker=idr_weight_linker,
            linker_crf=linker_crf,
            linker_gaussian_sigma=linker_gaussian_sigma,
            device=device,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            focal_positives_only=focal_positives_only,
        )
    
    def forward(
        self,
        disorder_logits: torch.Tensor,
        binding_logits: torch.Tensor,
        linker_logits: torch.Tensor,
        disorder_labels: torch.Tensor,
        binding_labels: torch.Tensor,
        linker_labels: torch.Tensor,
        mask: torch.Tensor,
        binding_mask: torch.Tensor = None,
        linker_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        loss_dict = {
            'disorder': torch.tensor(0.0, device=self.device),
            'binding': torch.tensor(0.0, device=self.device),
            'linker': torch.tensor(0.0, device=self.device)
        }
        
        # Phase 1 Disorder
        disorder_loss = self.phase1_loss(disorder_logits, disorder_labels, mask)
        loss_dict['disorder'] = disorder_loss.detach()
        
        # Phase 2 Function
        in_phase2 = (binding_logits is not None) or (linker_logits is not None)
        
        if in_phase2:
            p2_loss, p2_dict = self.phase2_loss(
                binding_logits, linker_logits, 
                binding_labels, linker_labels, 
                mask, 
                disorder_labels=disorder_labels,
                binding_mask=binding_mask,
                linker_mask=linker_mask
            )
            # Update dict
            loss_dict['total'] = p2_loss.detach()
            if 'binding' in p2_dict: loss_dict['binding'] = p2_dict['binding']
            if 'linker' in p2_dict: loss_dict['linker'] = p2_dict['linker']
            
            return p2_loss, loss_dict
        else:
            loss_dict['total'] = disorder_loss.detach()
            return disorder_loss, loss_dict

def compute_class_weights(
    dataset
) -> Tuple[float, float, float]:

    disorder_pos = 0
    disorder_neg = 0
    binding_pos = 0
    binding_neg = 0
    linker_pos = 0
    linker_neg = 0
    
    for i in range(len(dataset)):
        item = dataset[i]
        
        # Disorder counts
        disorder_pos += item['disorder_labels'].sum().item()
        disorder_neg += (1 - item['disorder_labels']).sum().item()
        
        # Binding: use combined binding label at index 4
        binding_pos += item['function_labels'][:, 4].sum().item()
        binding_neg += (1 - item['function_labels'][:, 4]).sum().item()
        
        # Linker: use linker label at index 5
        linker_pos += item['function_labels'][:, 5].sum().item()
        linker_neg += (1 - item['function_labels'][:, 5]).sum().item()
    
    # Compute weights (neg/pos ratio)
    disorder_weight = disorder_neg / max(disorder_pos, 1)
    binding_weight = binding_neg / max(binding_pos, 1)
    linker_weight = linker_neg / max(linker_pos, 1)
    
    # Clamp extremely high weights to prevent instability
    disorder_weight = min(disorder_weight, 30.0)
    binding_weight = min(binding_weight, 30.0)
    linker_weight = min(linker_weight, 30.0)
    
    logger.info(f"Disorder class weight: {disorder_weight:.3f}")
    logger.info(f"Binding class weight: {binding_weight:.3f}")
    logger.info(f"Linker class weight: {linker_weight:.3f}")
    
    return disorder_weight, binding_weight, linker_weight