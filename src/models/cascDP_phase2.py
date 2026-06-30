import torch
import torch.nn as nn
from typing import List, Optional
import math
import logging
from .cascDP_phase1 import cascDP_Phase1
from .context_modules import BiGRUContext, BiLSTMContext, CNNHead, ASPPBlock
from .fusion_modules import CrossAttentionFusion

logger = logging.getLogger(__name__)

class cascDP_Phase2(nn.Module):
    """
    Attaches binding and linker prediction heads to a trained Phase 1 model
    Phase 1 model is frozen, providing stable disorder features
    """
    
    BINDING_TYPES = ['Protein_binding', 'Nucleic_acid_binding', 'Ion_binding', 'Lipid_binding']
    NUM_BINDING_TYPES = 4  
    VALID_CONTEXT_TYPES = {"bigru", "bilstm"}

    @classmethod
    def _validate_context_type(cls, context_type: str, field_name: str = "context_type") -> str:
        if context_type not in cls.VALID_CONTEXT_TYPES:
            allowed = ", ".join(sorted(cls.VALID_CONTEXT_TYPES))
            raise ValueError(f"Unknown Phase 2 {field_name}: {context_type!r}. Expected one of: {allowed}")
        return context_type

    @classmethod
    def resolve_context_types(cls, model_config: dict):
        base_context = model_config.get('phase2_context_type', 'bigru')
        binding_context = model_config.get('binding_context_type', base_context)
        linker_context = model_config.get('linker_context_type', base_context)
        return (
            cls._validate_context_type(base_context, 'phase2_context_type'),
            cls._validate_context_type(binding_context, 'binding_context_type'),
            cls._validate_context_type(linker_context, 'linker_context_type'),
        )

    @staticmethod
    def _build_context(context_type: str, dim: int, dropout: float = 0.5):
        # dim is both input and output dim
        if context_type == "bilstm":
            return BiLSTMContext(in_channels=dim, hidden_channels=256, num_layers=2, dropout=dropout)
        if context_type == "bigru":
            return BiGRUContext(in_channels=dim, hidden_channels=256, num_layers=2, dropout=dropout)
        raise ValueError(f"Unknown Phase 2 context_type: {context_type!r}. Expected 'bigru' or 'bilstm'.")

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
        binding_combined: bool = False,
        binding_head_type: str = "cnn",
        binding_priors: Optional[List[float]] = None,
        binding_combined_prior: float = 0.1299,
        linker_prior: float = 0.0218,
    ):
        super().__init__()

        self.device = device
        self.phase1 = phase1_model
        self.hidden_dim = phase1_model.hidden_dim
        self.context_type = self._validate_context_type(context_type, 'context_type')
        self.binding_context_type = self._validate_context_type(
            binding_context_type or self.context_type, 'binding_context_type'
        )
        self.linker_context_type = self._validate_context_type(
            linker_context_type or self.context_type, 'linker_context_type'
        )
        self.use_binding_head = use_binding_head
        self.use_linker_head = use_linker_head
        self.binding_combined = binding_combined
        self.binding_head_type = binding_head_type
        self.dropout = dropout

        # Freeze Phase 1 (backbone + disorder)
        self.phase1.freeze()
        logger.info("Phase 2: Phase 1 model frozen")

        # Phase 1's ASPP output dim feeds the per-head disorder projection
        disorder_dim = phase1_model.aspp_out_dim  # 512
        binding_linker_input_dim = self.hidden_dim

        # Binding head
        if use_binding_head:
            binding_dim = 512
            # Downsize sequences and disorder features to bottleneck dimensions immediately
            self.binding_seq_proj = nn.Linear(self.hidden_dim, binding_dim)
            self.binding_disorder_proj = nn.Linear(disorder_dim, binding_dim)
            self.binding_disorder_proj_norm = nn.LayerNorm(binding_dim)
            self.binding_fusion = CrossAttentionFusion(
                sequence_dim=binding_dim,
                disorder_dim=binding_dim,
                num_heads=8,
                dropout=dropout
            )
            self.binding_esm_norm = nn.LayerNorm(binding_dim)
            logger.info("Phase 2: Bottlenecked Cross-Attention Fusion for Binding (Q=BiGRU, K/V=ESM) + ESM+disorder residual")

            self.binding_gru = self._build_context(self.binding_context_type, binding_dim, dropout=dropout)
            
            self.binding_self_attention = nn.MultiheadAttention(
                embed_dim=binding_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.binding_attn_norm = nn.LayerNorm(binding_dim)
            
            binding_output_dim = 1 if self.binding_combined else self.NUM_BINDING_TYPES
            self.binding_output_dim = binding_output_dim

            if self.binding_head_type == "aspp":
                # Multi-scale dilated context; tighter dilations (2,4,6) than the
                # linker head (2,4,8) to preserve locality for short binding motifs.
                self.binding_head = ASPPBlock(
                    in_dim=binding_dim,
                    out_dim=256,
                    dropout=dropout,
                    dilations=(2, 4, 6)
                )
                self.binding_final = nn.Linear(256, binding_output_dim)
                final_bias_layer = self.binding_final
            elif self.binding_head_type == "cnn":
                self.binding_head = CNNHead(
                    input_dim=binding_dim,
                    hidden_dims=[128, 64],
                    output_dim=binding_output_dim,
                    dropout=dropout,
                    dilation=2
                )
                self.binding_final = None
                final_bias_layer = self.binding_head.final
            else:
                raise ValueError(
                    f"Unknown binding_head_type: {self.binding_head_type!r}. Expected 'cnn' or 'aspp'."
                )

            # Bias init from class priors
            if self.binding_combined:
                bias_val = -math.log((1 - binding_combined_prior) / binding_combined_prior)
                torch.nn.init.constant_(final_bias_layer.bias, bias_val)
                logger.info(f"Phase 2: Initialized binding head (cross-attn -> {self.binding_context_type} -> self-attn -> {self.binding_head_type.upper()} -> 1 combined output, prior={binding_combined_prior:.4f})")
            else:
                # Multi-label: one output per binding type (Protein/Nucleic/Ion/Lipid)
                binding_priors = binding_priors or [0.1073, 0.0241, 0.0162, 0.0109]
                if len(binding_priors) != self.NUM_BINDING_TYPES:
                    raise ValueError(f"binding_priors must contain {self.NUM_BINDING_TYPES} values")
                for idx, prior in enumerate(binding_priors):
                    bias_val = -math.log((1 - prior) / prior)
                    torch.nn.init.constant_(final_bias_layer.bias[idx], bias_val)
                logger.info(f"Phase 2: Initialized binding head (cross-attn -> {self.binding_context_type} -> self-attn -> {self.binding_head_type.upper()} -> {self.NUM_BINDING_TYPES} multi-label outputs, priors={binding_priors})")
        
        # Linker head
        if use_linker_head:
            linker_dim = 256
            
            # Downsize sequences and disorder features to bottleneck dimensions immediately
            self.linker_seq_proj = nn.Linear(self.hidden_dim, linker_dim)
            self.linker_disorder_proj = nn.Linear(disorder_dim, linker_dim)
            self.linker_disorder_proj_norm = nn.LayerNorm(linker_dim)

            self.linker_fusion = CrossAttentionFusion(
                sequence_dim=linker_dim,
                disorder_dim=linker_dim,
                num_heads=8,
                dropout=dropout
            )
            self.linker_esm_norm = nn.LayerNorm(linker_dim)
            logger.info("Phase 2: Bottlenecked Cross-Attention Fusion for Linker")

            self.linker_gru = self._build_context(self.linker_context_type, linker_dim, dropout=dropout)

            linker_output_dim = 1
            self.linker_head = ASPPBlock(
                in_dim=linker_dim,
                out_dim=256,
                dropout=dropout,
                dilations=(2, 4, 8)
            )
            self.linker_final = nn.Linear(256, linker_output_dim)
            
            linker_bias = -math.log((1 - linker_prior) / linker_prior)
            torch.nn.init.constant_(self.linker_final.bias, linker_bias)
            logger.info(f"Phase 2: Initialized linker head (cross-attn -> {self.linker_context_type} -> ASPP -> Linear, bias={linker_bias:.4f}, prior={linker_prior:.4f})")
        
        self.to(device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Phase 2: {trainable:,} / {total:,} trainable parameters ({100*trainable/total:.2f}%)")

    @property
    def binding_output_layer(self) -> Optional[nn.Module]:
        # Final logit-producing Linear for the binding head, regardless of head type.
        if not getattr(self, 'use_binding_head', False):
            return None
        if getattr(self, 'binding_final', None) is not None:
            return self.binding_final
        return self.binding_head.final
    
    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        sequences: Optional[List[str]] = None,
        return_attention: bool = False
    ):
        # Force Phase 1 to eval mode
        self.phase1.eval()
        
        attention_dict = {} if return_attention else None

        # Get disorder prediction and embeddings from Phase 1 (frozen)
        with torch.no_grad():
            # Get raw ESM embeddings from Phase 1
            sequence_embeddings = self.phase1.get_embeddings(
                embeddings=embeddings,
                sequences=sequences
            )
            # disorder_features: 512-dim ASPP features for Phase 2 cascading
            if return_attention:
                disorder_features, disorder_logits, esm_cross_attn_weights = self.phase1.get_disorder_features(
                    sequence_embeddings, return_attention=True
                )
                attention_dict['phase1_cross_attn'] = esm_cross_attn_weights
            else:
                disorder_features, disorder_logits = self.phase1.get_disorder_features(
                    sequence_embeddings
                )
        
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
                    need_weights=True
                )
                attention_dict['binding_cross_attn'] = binding_cross_attn_weights
            else:
                binding_cross = self.binding_fusion(
                    sequence_embeddings=binding_bigru,
                    disorder_features=binding_seq
                )
            binding_combined = self.binding_esm_norm(binding_cross + binding_seq + binding_proj)

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
                    need_weights=True
                )
                attention_dict['linker_cross_attn'] = linker_cross_attn_weights
            else:
                linker_cross = self.linker_fusion(
                    sequence_embeddings=linker_bigru,
                    disorder_features=linker_seq
                )
            linker_features = self.linker_esm_norm(linker_cross + linker_seq + linker_proj)
        
        if self.use_binding_head:
            # Self-attention on binding features for global context
            if return_attention:
                binding_attn_out, binding_self_attn_weights = self.binding_self_attention(
                    binding_combined, binding_combined, binding_combined,
                    need_weights=True,
                    average_attn_weights=False
                )
                attention_dict['binding_self_attn'] = binding_self_attn_weights
            else:
                binding_attn_out, _ = self.binding_self_attention(
                    binding_combined, binding_combined, binding_combined,
                    need_weights=False
                )
            binding_combined = self.binding_attn_norm(binding_combined + binding_attn_out)
            binding_features = binding_combined
        
        binding_logits = None
        if self.use_binding_head:
            binding_out = self.binding_head(binding_features)
            if self.binding_final is not None:
                binding_logits = self.binding_final(binding_out)
                if self.binding_output_dim == 1:
                    binding_logits = binding_logits.squeeze(-1)
            else:
                binding_logits = binding_out
        
        linker_logits = None
        if self.use_linker_head:
             linker_out = self.linker_head(linker_features)
             linker_logits = self.linker_final(linker_out)
        
        if return_attention:
            return disorder_logits, binding_logits, linker_logits, attention_dict
            
        return disorder_logits, binding_logits, linker_logits
    
    @classmethod
    def from_phase1_checkpoint(
        cls,
        checkpoint_path: str,
        phase1_model: cascDP_Phase1,
        device: str = "cuda",
        **kwargs
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        phase1_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Loaded Phase 1 checkpoint from {checkpoint_path}")
    
        phase2_model = cls(phase1_model=phase1_model, device=device, **kwargs)
        return phase2_model

    @classmethod
    def from_separate_checkpoints(
        cls,
        binding_ckpt_path: str,
        linker_ckpt_path: str,
        phase1_model: cascDP_Phase1,
        device: str = "cuda",
        **kwargs
    ):
        """
        Build a unified Phase 2 model by merging weights from two separately trained
        checkpoints: one trained with only the binding head, one with only the linker head.

        Binding-prefixed keys come from the binding checkpoint; linker-prefixed keys
        come from the linker checkpoint.  Shared weights (phase1.*) are identical in
        both (Phase 1 is frozen) and taken from the binding checkpoint.

        Args:
            binding_ckpt_path: Checkpoint trained with use_binding_head=True, use_linker_head=False
            linker_ckpt_path:  Checkpoint trained with use_binding_head=False, use_linker_head=True
            phase1_model:      Initialised cascDP_Phase1 instance (weights loaded from binding ckpt)
            device:            Device to place the model on
            **kwargs:          Architecture kwargs forwarded to __init__
                               (context_type, …).
                               use_binding_head and use_linker_head are forced to True.

        Returns:
            cascDP_Phase2 with both heads populated from their respective checkpoints.
        """
        binding_ckpt = torch.load(binding_ckpt_path, map_location=device, weights_only=False)
        linker_ckpt  = torch.load(linker_ckpt_path,  map_location=device, weights_only=False)

        # Force both heads enabled in the unified model
        kwargs['use_binding_head'] = True
        kwargs['use_linker_head']  = True

        model = cls(phase1_model=phase1_model, device=device, **kwargs)

        binding_state = binding_ckpt['model_state_dict']
        linker_state  = linker_ckpt['model_state_dict']
        merged        = model.state_dict()

        for key in merged:
            if key.startswith('linker_'):
                # Linker-specific tensors come from linker checkpoint
                if key in linker_state:
                    merged[key] = linker_state[key]
                else:
                    logger.warning(f"Key '{key}' not found in linker checkpoint — keeping random init")
            elif key.startswith('binding_'):
                # Binding-specific tensors come from binding checkpoint
                if key in binding_state:
                    merged[key] = binding_state[key]
                else:
                    logger.warning(f"Key '{key}' not found in binding checkpoint — keeping random init")
            else:
                # Shared weights (phase1.*, disorder_proj*, etc.)
                # These are frozen during Phase 2 training, so identical in both checkpoints
                if key in binding_state:
                    merged[key] = binding_state[key]
                elif key in linker_state:
                    merged[key] = linker_state[key]
                else:
                    logger.warning(f"Key '{key}' not found in either checkpoint — keeping random init")

        model.load_state_dict(merged, strict=True)
        logger.info(
            f"Merged binding checkpoint ({binding_ckpt_path}) + "
            f"linker checkpoint ({linker_ckpt_path}) into unified Phase 2 model"
        )
        return model
