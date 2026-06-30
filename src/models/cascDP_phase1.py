import torch
import torch.nn as nn
from typing import List, Optional
import math
import logging
from .backbone import ProteinBackbone
from .context_modules import BiGRUContext, ASPPBlock, BiLSTMContext, CNNHead
from .fusion_modules import CrossAttentionFusion

logger = logging.getLogger(__name__)

class cascDP_Phase1(nn.Module):    
    # Trains backbone + disorder head on disorder annotations
    # This model serves as the foundation for Phase 2 function prediction
    
    def __init__(
        self,
        backbone: ProteinBackbone,
        device: str = "cuda",
        context_type: str = "bigru",
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        disorder_prior: float = 0.1159,
        fusion_type: str = "sum",
    ):
        super().__init__()
        
        self.device = device
        self.backbone_wrapper = backbone
        self.backbone = backbone.get_model()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Phase 1: Backbone frozen")

        self.hidden_dim = backbone.get_hidden_dim()
        self.model_name = getattr(backbone, 'model_name', 'custom_backbone')
        
        self.context_type = context_type
        self.dropout = dropout

        # Local context modeling
        if self.context_type == "bilstm":
            self.local_context = BiLSTMContext(
                in_channels=self.hidden_dim,
                hidden_channels=384,
                num_layers=2,
                dropout=dropout
            )
            logger.info("Phase 1: Using Bi-LSTM context")
        else:
            self.local_context = BiGRUContext(
                in_channels=self.hidden_dim,
                hidden_channels=384,
                num_layers=2,
                dropout=dropout
            )
            logger.info("Phase 1: Using Bi-GRU context (2 layers)")
        
        # Disorder head
        # self.disorder_head_stage1 = CNNHead(
        #     input_dim=self.hidden_dim,
        #     hidden_dims=[512, 256, 128],
        #     output_dim=64,
        #     dropout=0.5
        # )
        
        # Use ASPP instead of CNNHead for disorder feature extraction
        self.aspp_out_dim = 512
        self.disorder_head_stage1 = ASPPBlock(
            in_dim=self.hidden_dim,
            out_dim=self.aspp_out_dim,
            dropout=dropout,
            dilations=(3, 12, 24)
        )
        
        # Cross-attention: BiLSTM/BiGRU output (Q) attends to raw ESM embeddings (K/V)
        self.esm_cross_attn = CrossAttentionFusion(
            sequence_dim=self.hidden_dim,
            disorder_dim=self.hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        self.esm_cross_attn_norm = nn.LayerNorm(self.hidden_dim)

        # Fusion strategy for combining cross-attention, local context, and raw ESM signals.
        #   "sum"   — LN(cross + raw + local), no extra parameters.
        #   "gate"  — per-residue learned gate (Variant A, bottlenecked):
        #             g = sigmoid(MLP([cross; local; raw]));  enriched = LN(g*cross + (1-g)*(raw+local)).
        #   "alpha" — learned per-channel static blend (Variant B):
        #             enriched = LN(alpha*cross + (1-alpha)*local) + raw.
        self.fusion_type = fusion_type
        if self.fusion_type == "gate":
            gate_hidden = max(1, self.hidden_dim // 4)
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.hidden_dim),
            )
            # Bias the final gate layer toward 0 logits -> g≈0.5 (balanced) at init.
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)
            logger.info(f"Phase 1: Gated cross-attn fusion (Variant A, bottleneck dim={gate_hidden})")
        elif self.fusion_type == "alpha":
            # Per-channel blend weight in [0,1] via sigmoid; init logit 0 -> alpha≈0.5.
            self.fusion_alpha_logit = nn.Parameter(torch.zeros(self.hidden_dim))
            logger.info("Phase 1: Learned per-channel alpha fusion (Variant B)")
        else:
            self.fusion_type = "sum"
            logger.info("Phase 1: Sum fusion (cross + raw + local)")
        
        # 3-layer funnel MLP before final projection
        self.disorder_mlp = nn.Sequential(
            nn.Linear(self.aspp_out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.disorder_initial = nn.Linear(128, 1)
        # Initialize disorder bias from dataset prior
        disorder_bias = -math.log((1 - disorder_prior) / disorder_prior)
        torch.nn.init.constant_(self.disorder_initial.bias, disorder_bias)
        logger.info(f"Phase 1: Initialized disorder bias to {disorder_bias:.4f} (prior={disorder_prior:.4f})")

        # Output dim of the MLP funnel — used by Phase 2 MLP-cascade experiments
        self.mlp_out_dim = self.disorder_initial.in_features  # 128

        self.to(device)
        logger.info("Phase 1 model created")
    
    def _run_disorder_pipeline(self, sequence_embeddings: torch.Tensor, return_attention: bool = False):
        local_features = self.local_context(sequence_embeddings)

        # Cross-attention: Q=BiLSTM output, K/V=raw ESM
        if return_attention:
            cross_out, esm_cross_attn_weights = self.esm_cross_attn(
                sequence_embeddings=local_features,
                disorder_features=sequence_embeddings,
                need_weights=True
            )
        else:
            cross_out = self.esm_cross_attn(
                sequence_embeddings=local_features,
                disorder_features=sequence_embeddings
            )
            esm_cross_attn_weights = None
            
        if self.fusion_type == "gate":
            # Variant A: per-residue learned gate over cross-attention vs. (raw + local).
            g = torch.sigmoid(self.fusion_gate(
                torch.cat([cross_out, local_features, sequence_embeddings], dim=-1)
            ))
            enriched = self.esm_cross_attn_norm(
                g * cross_out + (1.0 - g) * (sequence_embeddings + local_features)
            )
        elif self.fusion_type == "alpha":
            # Variant B: learned per-channel static blend; raw ESM added as residual.
            a = torch.sigmoid(self.fusion_alpha_logit)
            enriched = self.esm_cross_attn_norm(
                a * cross_out + (1.0 - a) * local_features
            ) + sequence_embeddings
        else:
            enriched = self.esm_cross_attn_norm(cross_out + sequence_embeddings + local_features)

        aspp_features = self.disorder_head_stage1(enriched)
        disorder_features = self.disorder_mlp(aspp_features)
        disorder_logits = self.disorder_initial(disorder_features)

        if return_attention:
            return aspp_features, disorder_logits, esm_cross_attn_weights
        return aspp_features, disorder_logits

    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
        return_attention: bool = False
    ):
        if embeddings is not None:
            sequence_embeddings = embeddings
        elif sequences is not None:
            sequence_embeddings = self._encode_sequences(sequences)
        else:
            raise ValueError("Either 'sequences' or 'embeddings' must be provided")

        if return_attention:
            _, disorder_logits, esm_cross_attn_weights = self._run_disorder_pipeline(
                sequence_embeddings, return_attention=True
            )
            return disorder_logits, esm_cross_attn_weights
            
        _, disorder_logits = self._run_disorder_pipeline(sequence_embeddings)
        return disorder_logits

    def _encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise ImportError(
                "Sequence encoding requires the esm package. "
                "Precomputed-embedding submission inference does not use this path."
            ) from exc

        embeddings_list = []
        model = self.backbone

        # Handle LoRA wrappers
        if not hasattr(model, 'encode') and hasattr(model, 'base_model'):
            model = model.base_model
        if not hasattr(model, 'encode') and hasattr(model, 'model'):
            model = model.model

        # When training with an unfrozen backbone, call forward() directly so
        # gradients flow into LoRA adapters.  model.logits() wraps the forward
        # pass in torch.no_grad(), which starves the backbone of gradients.
        grad_enabled = self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )

        for seq in sequences:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)

            if grad_enabled:
                tokens = protein_tensor.sequence.unsqueeze(0)
                output = model(sequence_tokens=tokens)
                emb = output.embeddings
            else:
                output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                emb = output.embeddings

            if not isinstance(emb, torch.Tensor):
                emb = torch.from_numpy(emb)
            if emb.dim() == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)

            # Backbone runs in bf16 on GPU; downstream heads are fp32
            if emb.dtype != torch.float32:
                emb = emb.float()

            # Remove BOS/EOS tokens if present
            if emb.shape[0] == len(seq) + 2:
                emb = emb[1:-1]

            embeddings_list.append(emb)

        # Pad to max length
        max_len = max(e.shape[0] for e in embeddings_list)
        padded_embeddings = []

        for emb in embeddings_list:
            seq_len = emb.shape[0]
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, self.hidden_dim, device=emb.device)
                emb = torch.cat([emb, padding], dim=0)
            padded_embeddings.append(emb)

        return torch.stack(padded_embeddings, dim=0).to(self.device)
    
    def get_embeddings(
        self,
        sequences: Optional[List[str]] = None,
        embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get raw ESM embeddings. Used by Phase 2 model
        if embeddings is not None:
            return embeddings
        elif sequences is not None:
            return self._encode_sequences(sequences)
        else:
            raise ValueError("Either 'sequences' or 'embeddings' must be provided")
    
    def get_disorder_features(
        self,
        raw_embeddings: torch.Tensor,
        return_attention: bool = False
    ):
        """
        Runs full Phase 1 disorder pipeline on raw ESM embeddings
        Flow: BiLSTM -> CrossAttn(Q=BiLSTM, K/V=ESM) -> +(ESM+BiLSTM) residuals -> ASPP(512) -> MLP(512->256->128) -> logits
        Returns ASPP features (512-dim) for Phase 2 cascading, disorder logits, and optionally attention weights.
        """
        return self._run_disorder_pipeline(raw_embeddings, return_attention=return_attention)

    def get_mlp_disorder_features(
        self,
        raw_embeddings: torch.Tensor,
        return_attention: bool = False
    ):
        if return_attention:
            aspp_features, disorder_logits, weights = self._run_disorder_pipeline(
                raw_embeddings, return_attention=True
            )
            mlp_hidden = self.disorder_mlp(aspp_features)
            return mlp_hidden, disorder_logits, weights
        aspp_features, disorder_logits = self._run_disorder_pipeline(raw_embeddings)
        mlp_hidden = self.disorder_mlp(aspp_features)
        return mlp_hidden, disorder_logits

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Phase 1 model frozen")
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Phase 1 model unfrozen")
