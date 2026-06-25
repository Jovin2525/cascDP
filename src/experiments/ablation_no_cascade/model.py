"""
src/experiments/ablation_no_cascade/model.py

No-disorder-cascade ablation model.

Differences from cascDP_Phase2:
  - Phase 1 is NOT instantiated.
  - The LoRA-fine-tuned backbone is loaded directly from the Phase 1 checkpoint.
  - Zero disorder features are fed to the function heads — raw ESM embeddings only.
  - Function head architecture is identical to Phase 2 (BiGRU -> self-attn ->
    CNNHead for binding; BiGRU -> ASPP for linker).

To load from a Phase 1 checkpoint (recommended — reuses LoRA weights):
    model = cascDP_Ablation1.from_phase1_checkpoint(
        checkpoint_path="checkpoints/phase1/best_model.pt",
        backbone=create_backbone(...),   # same config as Phase 1
        device="cuda",
    )
"""

import math
import logging
from typing import List, Optional

import torch
import torch.nn as nn

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from ...models.backbone import ProteinBackbone
from ...models.context_modules import (
    BiGRUContext,
    BiLSTMContext,
    CNNHead,
    ASPPBlock,
)
from ...models.fusion_modules import CrossAttentionFusion

logger = logging.getLogger(__name__)

class cascDP_Ablation1(nn.Module):
    """
    Ablation model: binding + linker heads on raw ESM embeddings only.

    Phase 1's disorder pipeline (BiGRU -> CrossAttn -> ASPP -> MLP) is entirely
    absent.  The backbone (ESM-C + LoRA) is the only shared component with
    Phase 2, ensuring that any performance difference is attributable to the
    disorder cascade alone.

    Args:
        backbone:         ProteinBackbone instance (e.g. ESMCBackbone).
                          Should already have LoRA applied if desired.
        device:           Device string.
        context_type:     "bigru" | "bilstm".
        use_binding_head: Include binding prediction head.
        use_linker_head:  Include linker prediction head.
        freeze_backbone:  Freeze backbone weights.  Set True to mirror the
                          Phase 2 setup (frozen Phase 1 backbone).
    """

    BINDING_TYPES = ["Protein_binding", "Nucleic_acid_binding", "Ion_binding", "Lipid_binding"]
    NUM_BINDING_TYPES = 4

    @staticmethod
    def _build_context(context_type: str, dim: int, dropout: float = 0.5) -> nn.Module:
        if context_type == "bilstm":
            return BiLSTMContext(in_channels=dim, hidden_channels=256, num_layers=1, dropout=dropout)
        else:  # bigru (default)
            return BiGRUContext(in_channels=dim, hidden_channels=256, num_layers=1, dropout=dropout)
        
    def __init__(
        self,
        backbone: ProteinBackbone,
        device: str = "cuda",
        context_type: str = "bigru",
        binding_context_type: Optional[str] = None,
        linker_context_type: Optional[str] = None,
        use_binding_head: bool = True,
        use_linker_head: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
        binding_combined: bool = False,
        binding_head_type: str = "cnn",
    ):
        super().__init__()

        if not use_binding_head and not use_linker_head:
            raise ValueError("At least one of use_binding_head or use_linker_head must be True")

        self.device = device
        self.backbone_wrapper = backbone
        self.backbone = backbone.get_model()   # ESM model (may have LoRA adapters)
        self.hidden_dim = backbone.get_hidden_dim()
        self.context_type = context_type
        self.binding_context_type = binding_context_type or context_type
        self.linker_context_type = linker_context_type or context_type
        self.use_binding_head = use_binding_head
        self.use_linker_head = use_linker_head
        self.binding_combined = binding_combined
        self.binding_head_type = binding_head_type

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Ablation1: backbone frozen")

        if use_binding_head:
            binding_dim = 512
            self.binding_seq_proj = nn.Linear(self.hidden_dim, binding_dim)
            self.binding_context_proj = nn.Linear(self.hidden_dim, binding_dim)
            self.binding_context_proj_norm = nn.LayerNorm(binding_dim)
            self.binding_fusion = CrossAttentionFusion(
                sequence_dim=binding_dim,
                disorder_dim=binding_dim,
                num_heads=8,
                dropout=dropout,
            )
            self.binding_esm_norm = nn.LayerNorm(binding_dim)
            self.binding_gru = self._build_context(self.binding_context_type, binding_dim, dropout=dropout)
        if use_linker_head:
            linker_dim = 256
            self.linker_seq_proj = nn.Linear(self.hidden_dim, linker_dim)
            self.linker_context_proj = nn.Linear(self.hidden_dim, linker_dim)
            self.linker_context_proj_norm = nn.LayerNorm(linker_dim)
            self.linker_fusion = CrossAttentionFusion(
                sequence_dim=linker_dim,
                disorder_dim=linker_dim,
                num_heads=8,
                dropout=dropout,
            )
            self.linker_esm_norm = nn.LayerNorm(linker_dim)
            self.linker_gru = self._build_context(self.linker_context_type, linker_dim, dropout=dropout)
        logger.info(
            f"Ablation1: binding={self.binding_context_type}, linker={self.linker_context_type} "
            "Phase2-matched sequence-only heads"
        )

        if use_binding_head:
            self.binding_self_attention = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.binding_attn_norm = nn.LayerNorm(512)

        if use_binding_head:
            binding_output_dim = 1 if self.binding_combined else self.NUM_BINDING_TYPES
            self.binding_output_dim = binding_output_dim

            if self.binding_head_type == "aspp":
                self.binding_head = ASPPBlock(
                    in_dim=512,
                    out_dim=256,
                    dropout=dropout,
                    dilations=(2, 4, 6),
                )
                self.binding_final = nn.Linear(256, binding_output_dim)
                final_bias_layer = self.binding_final
            elif self.binding_head_type == "cnn":
                self.binding_head = CNNHead(
                    input_dim=512,
                    hidden_dims=[128, 64],
                    output_dim=binding_output_dim,
                    dropout=dropout,
                    dilation=2,
                )
                self.binding_final = None
                final_bias_layer = self.binding_head.final
            else:
                raise ValueError("binding_head_type must be 'cnn' or 'aspp'")

            if self.binding_combined:
                prior = 0.1299
                nn.init.constant_(final_bias_layer.bias, -math.log((1 - prior) / prior))
            else:
                for idx, prior in enumerate([0.1073, 0.0241, 0.0162, 0.0109]):
                    nn.init.constant_(final_bias_layer.bias[idx], -math.log((1 - prior) / prior))

        if use_linker_head:
            self.linker_head = ASPPBlock(in_dim=256, out_dim=256, dropout=dropout, dilations=(2, 4, 8))
            self.linker_final = nn.Linear(256, 1)
            nn.init.constant_(
                self.linker_final.bias,
                -math.log((1 - 0.0218) / 0.0218),
            )

        self.to(device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Ablation1: {trainable:,} / {total:,} trainable params "
            f"({100 * trainable / total:.2f}%)"
        )

    def _encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode raw sequences into ESM embeddings.
        Mirrors cascDP_Phase1._encode_sequences exactly.
        """
        model = self.backbone
        # Unwrap LoRA / PEFT wrappers if needed
        if not hasattr(model, "encode") and hasattr(model, "base_model"):
            model = model.base_model
        if not hasattr(model, "encode") and hasattr(model, "model"):
            model = model.model

        embeddings_list = []
        for seq in sequences:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            emb = output.embeddings
            if not isinstance(emb, torch.Tensor):
                emb = torch.from_numpy(emb)
            if emb.dim() == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            # Remove BOS/EOS tokens if present
            if emb.shape[0] == len(seq) + 2:
                emb = emb[1:-1]
            embeddings_list.append(emb)

        max_len = max(e.shape[0] for e in embeddings_list)
        padded = []
        for emb in embeddings_list:
            if emb.shape[0] < max_len:
                pad = torch.zeros(max_len - emb.shape[0], self.hidden_dim, device=emb.device)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        return torch.stack(padded, dim=0).to(self.device)

    def get_embeddings(
        self,
        embeddings: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if embeddings is not None:
            return embeddings
        if sequences is not None:
            return self._encode_sequences(sequences)
        raise ValueError("Either 'embeddings' or 'sequences' must be provided")

    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
    ):
        """
        Forward pass — binding and linker from raw ESM embeddings.

        No disorder cascade is applied.

        Returns:
            Tuple (None, binding_logits | None, linker_logits | None)
            First element is always None for API compatibility with Phase 2 /
            CascadedLoss / AblationTrainer.
        """
        esm_emb = self.get_embeddings(embeddings=embeddings, sequences=sequences)

        binding_logits = None
        if self.use_binding_head:
            binding_seq = self.binding_seq_proj(esm_emb)
            binding_proj = self.binding_context_proj_norm(self.binding_context_proj(esm_emb))
            binding_bigru = binding_proj + self.binding_gru(binding_proj)
            binding_cross = self.binding_fusion(
                sequence_embeddings=binding_bigru,
                disorder_features=binding_seq,
            )
            binding_ctx = self.binding_esm_norm(binding_cross + binding_seq + binding_proj)
            attn_out, _ = self.binding_self_attention(
                binding_ctx, binding_ctx, binding_ctx, need_weights=False
            )
            binding_ctx = self.binding_attn_norm(binding_ctx + attn_out)
            binding_out = self.binding_head(binding_ctx)
            if self.binding_final is not None:
                binding_logits = self.binding_final(binding_out)
                if self.binding_output_dim == 1:
                    binding_logits = binding_logits.squeeze(-1)
            else:
                binding_logits = binding_out

        linker_logits = None
        if self.use_linker_head:
            linker_seq = self.linker_seq_proj(esm_emb)
            linker_proj = self.linker_context_proj_norm(self.linker_context_proj(esm_emb))
            linker_bigru = linker_proj + self.linker_gru(linker_proj)
            linker_cross = self.linker_fusion(
                sequence_embeddings=linker_bigru,
                disorder_features=linker_seq,
            )
            linker_ctx = self.linker_esm_norm(linker_cross + linker_seq + linker_proj)
            linker_feats = self.linker_head(linker_ctx)       # (B, L, 256)
            linker_logits = self.linker_final(linker_feats).squeeze(-1)  # (B, L)

        return None, binding_logits, linker_logits

    @classmethod
    def from_phase1_checkpoint(
        cls,
        checkpoint_path: str,
        backbone: ProteinBackbone,
        device: str = "cuda",
        **kwargs,
    ) -> "cascDP_Ablation1":
        """
        Build ablation model, loading LoRA-fine-tuned backbone from a Phase 1 checkpoint.

        Only the ``backbone.*`` keys from the checkpoint are applied to the
        backbone model.  All other Phase 1 keys (local_context, disorder head,
        MLP, CRF, …) are ignored — they do not exist in the ablation model.

        Args:
            checkpoint_path: Path to cascDP_Phase1 checkpoint (.pt file).
            backbone:        ProteinBackbone with LoRA already configured
                             (same r, alpha, target_modules as the checkpoint).
            device:          Device string.
            **kwargs:        Forwarded to __init__
                             (context_type, use_binding_head, use_linker_head,
                             freeze_backbone, …).

        Returns:
            cascDP_Ablation1 with LoRA-fine-tuned backbone weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]

        # Extract only the backbone sub-tree (keys: "backbone.<rest>")
        backbone_prefix = "backbone."
        backbone_state = {
            k[len(backbone_prefix):]: v
            for k, v in state.items()
            if k.startswith(backbone_prefix)
        }

        if backbone_state:
            missing, unexpected = backbone.get_model().load_state_dict(
                backbone_state, strict=False
            )
            logger.info(
                f"Ablation1: backbone weights loaded from {checkpoint_path} "
                f"({len(backbone_state)} tensors, "
                f"{len(missing)} missing, {len(unexpected)} unexpected)"
            )
        else:
            logger.warning(
                f"Ablation1: no 'backbone.*' keys found in {checkpoint_path} — "
                "backbone weights NOT loaded.  Check that LoRA config matches."
            )

        return cls(backbone=backbone, device=device, **kwargs)
