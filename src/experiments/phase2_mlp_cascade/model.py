"""
src/experiments/phase2_mlp_cascade/model.py

Phase 2 variant that cascades Phase 1's 128-dim MLP hidden features instead of
the 512-dim ASPP output.

Hypothesis:
    The MLP funnel (ASPP(512) -> Linear(256) -> Linear(128)) is the most
    disorder-discriminative representation in Phase 1 — it feeds directly into
    disorder_initial and is supervised end-to-end for binary disorder classification.
    Using this more distilled bottleneck as the cascade signal may provide a
    sharper gradient pathway and cleaner disorder encoding than the 512-dim ASPP.

Training:
    python -m src.cli.train_phase2_mlp_cascade \\
        --config configs/experiments/phase2_mlp_cascade/binding.yaml
    python -m src.cli.train_phase2_mlp_cascade \\
        --config configs/experiments/phase2_mlp_cascade/linker.yaml
"""

import math
import logging
from typing import List, Optional

import torch
import torch.nn as nn

from ...models.cascDP_phase1 import cascDP_Phase1
from ...models.cascDP_phase2 import cascDP_Phase2
from ...models.context_modules import BiGRUContext, BiLSTMContext, CNNHead, ASPPBlock
from ...models.fusion_modules import CrossAttentionFusion

logger = logging.getLogger(__name__)

# Minimum attention head dimension to avoid degenerate attention maps
_MIN_HEAD_DIM = 32


def _build_seq_proj(hidden_dim: int, cascade_dim: int) -> nn.Module:
    """
    Build the ESM -> cascade_dim projection.

    When cascade_dim is much smaller than hidden_dim (ratio ≥ 4), a two-stage
    funnel avoids an abrupt single-step bottleneck:
        hidden_dim -> hidden_dim//2  (GELU + LN)  -> cascade_dim
    Otherwise a single Linear is used.
    """
    if hidden_dim // cascade_dim >= 4:
        mid = max(cascade_dim * 2, hidden_dim // 2)
        return nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, cascade_dim),
        )
    return nn.Linear(hidden_dim, cascade_dim)


class cascDP_Phase2_MLPCascade(cascDP_Phase2):
    """
    Phase 2 variant: cascades 128-dim MLP hidden features from Phase 1 instead
    of 512-dim ASPP features.
    """

    def __init__(
        self,
        phase1_model: cascDP_Phase1,
        device: str = "cuda",
        context_type: str = "bigru",
        binding_context_type: Optional[str] = None,
        linker_context_type: Optional[str] = None,
        dropout: float = 0.2,
        use_binding_head: bool = True,
        use_linker_head: bool = True,
        cascade_dim: int = 512,
    ):
        # The parent __init__ reads phase1_model.aspp_out_dim to determine disorder_dim
        # for all disorder projection layers.  Temporarily replace it with mlp_out_dim
        # (128) so those layers are built with the correct input dimension.
        _saved_aspp_out_dim = phase1_model.aspp_out_dim
        phase1_model.aspp_out_dim = phase1_model.mlp_out_dim  # 128
        try:
            super().__init__(
                phase1_model=phase1_model,
                device=device,
                context_type=context_type,
                binding_context_type=binding_context_type,
                linker_context_type=linker_context_type,
                dropout=dropout,
                use_binding_head=use_binding_head,
                use_linker_head=use_linker_head,
            )
        finally:
            # Always restore, even if super().__init__() raises
            phase1_model.aspp_out_dim = _saved_aspp_out_dim

        self.cascade_dim = cascade_dim
        mlp_out = phase1_model.mlp_out_dim  # 128

        # If cascade_dim differs from the default 512 used by the parent, rebuild
        # all per-head layers so they operate at cascade_dim throughout
        if cascade_dim != 512:
            self._rebuild_head_layers(
                mlp_out=mlp_out,
                cascade_dim=cascade_dim,
                dropout=dropout,
            )
            self.to(device)
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logger.info(
                "Phase2_MLPCascade (cascade_dim=%d): %s / %s trainable params (%.2f%%)",
                cascade_dim, f"{trainable:,}", f"{total:,}", 100 * trainable / total,
            )

        logger.info(
            "Phase2_MLPCascade: MLP hidden (%d-dim) cascade -> %d-dim heads",
            mlp_out, cascade_dim,
        )

    def _rebuild_head_layers(
        self,
        mlp_out: int,
        cascade_dim: int,
        dropout: float,
    ):
        # Rebuild all Phase 2 head layers at cascade_dim
        num_heads = max(1, cascade_dim // _MIN_HEAD_DIM)

        if self.use_binding_head:
            self.binding_seq_proj = _build_seq_proj(self.hidden_dim, cascade_dim)
            self.binding_disorder_proj = nn.Linear(mlp_out, cascade_dim)
            self.binding_disorder_proj_norm = nn.LayerNorm(cascade_dim)
            self.binding_gru = self._build_context(
                self.binding_context_type, cascade_dim, dropout=dropout
            )
            self.binding_fusion = CrossAttentionFusion(
                sequence_dim=cascade_dim,
                disorder_dim=cascade_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.binding_esm_norm = nn.LayerNorm(cascade_dim)
            self.binding_self_attention = nn.MultiheadAttention(
                embed_dim=cascade_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.binding_attn_norm = nn.LayerNorm(cascade_dim)
            hidden1 = max(32, cascade_dim // 2)
            hidden2 = max(16, cascade_dim // 4)
            self.binding_head = CNNHead(
                input_dim=cascade_dim,
                hidden_dims=[hidden1, hidden2],
                output_dim=self.NUM_BINDING_TYPES,
                dropout=dropout,
                dilation=2,
            )
            # Re-initialise binding output bias from dataset priors
            binding_priors = [0.1073, 0.0241, 0.0162, 0.0109]
            for idx, prior in enumerate(binding_priors):
                bias_val = -math.log((1 - prior) / prior)
                nn.init.constant_(self.binding_head.final.bias[idx], bias_val)

        if self.use_linker_head:
            self.linker_seq_proj = _build_seq_proj(self.hidden_dim, cascade_dim)
            self.linker_disorder_proj = nn.Linear(mlp_out, cascade_dim)
            self.linker_disorder_proj_norm = nn.LayerNorm(cascade_dim)
            self.linker_gru = self._build_context(
                self.linker_context_type, cascade_dim, dropout=dropout
            )
            self.linker_fusion = CrossAttentionFusion(
                sequence_dim=cascade_dim,
                disorder_dim=cascade_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.linker_esm_norm = nn.LayerNorm(cascade_dim)
            linker_output_dim = 1
            self.linker_head = ASPPBlock(
                in_dim=cascade_dim,
                out_dim=cascade_dim,
                dropout=dropout,
                dilations=(2, 4, 8),
            )
            self.linker_final = nn.Linear(cascade_dim, linker_output_dim)
            linker_prior = 0.0218
            linker_bias = -math.log((1 - linker_prior) / linker_prior)
            nn.init.constant_(self.linker_final.bias, linker_bias)

    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
        return_attention: bool = False,
    ):
        """
        Identical to cascDP_Phase2.forward() except Phase 1 is queried via
        get_mlp_disorder_features() -> 128-dim MLP hidden instead of 512-dim ASPP.
        """
        self.phase1.eval()
        attention_dict = {} if return_attention else None

        with torch.no_grad():
            sequence_embeddings = self.phase1.get_embeddings(
                embeddings=embeddings,
                sequences=sequences,
            )

            if return_attention:
                disorder_features, disorder_logits, esm_cross_attn_weights = (
                    self.phase1.get_mlp_disorder_features(
                        sequence_embeddings, return_attention=True
                    )
                )
                attention_dict["phase1_cross_attn"] = esm_cross_attn_weights
            else:
                disorder_features, disorder_logits = (
                    self.phase1.get_mlp_disorder_features(sequence_embeddings)
                )

        # Binding head
        if self.use_binding_head:
            binding_seq = self.binding_seq_proj(sequence_embeddings)
            binding_proj = self.binding_disorder_proj_norm(
                self.binding_disorder_proj(disorder_features)
            )
            binding_bigru = binding_proj + self.binding_gru(binding_proj)
            if return_attention:
                binding_cross, binding_cross_attn_weights = self.binding_fusion(
                    sequence_embeddings=binding_bigru,
                    disorder_features=binding_seq,
                    need_weights=True,
                )
                attention_dict["binding_cross_attn"] = binding_cross_attn_weights
            else:
                binding_cross = self.binding_fusion(
                    sequence_embeddings=binding_bigru,
                    disorder_features=binding_seq,
                )
            binding_combined = self.binding_esm_norm(
                binding_cross + binding_seq + binding_proj
            )

        # Linker head
        if self.use_linker_head:
            linker_seq = self.linker_seq_proj(sequence_embeddings)
            linker_proj = self.linker_disorder_proj_norm(
                self.linker_disorder_proj(disorder_features)
            )
            linker_bigru = linker_proj + self.linker_gru(linker_proj)
            if return_attention:
                linker_cross, linker_cross_attn_weights = self.linker_fusion(
                    sequence_embeddings=linker_bigru,
                    disorder_features=linker_seq,
                    need_weights=True,
                )
                attention_dict["linker_cross_attn"] = linker_cross_attn_weights
            else:
                linker_cross = self.linker_fusion(
                    sequence_embeddings=linker_bigru,
                    disorder_features=linker_seq,
                )
            linker_combined = self.linker_esm_norm(
                linker_cross + linker_seq + linker_proj
            )
            linker_features = linker_combined

        # Binding self-attention
        if self.use_binding_head:
            if return_attention:
                binding_attn_out, binding_self_attn_weights = self.binding_self_attention(
                    binding_combined, binding_combined, binding_combined,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attention_dict["binding_self_attn"] = binding_self_attn_weights
            else:
                binding_attn_out, _ = self.binding_self_attention(
                    binding_combined, binding_combined, binding_combined,
                    need_weights=False,
                )
            binding_combined = self.binding_attn_norm(binding_combined + binding_attn_out)
            binding_features = binding_combined

        # Predictions
        binding_logits = self.binding_head(binding_features) if self.use_binding_head else None

        linker_logits = None
        if self.use_linker_head:
            linker_out = self.linker_head(linker_features)
            linker_logits = self.linker_final(linker_out)

        if return_attention:
            return disorder_logits, binding_logits, linker_logits, attention_dict
        return disorder_logits, binding_logits, linker_logits
