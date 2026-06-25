import torch
import torch.nn as nn
import random
import logging
from .cascDP_phase1 import cascDP_Phase1

logger = logging.getLogger(__name__)

class cascDP_Phase1Recycle(cascDP_Phase1):
    """
    Phase 1 disorder prediction with iterative recycling.

    Recycles the `enriched` trunk state (1152-dim, post cross-attn norm) back into
    the BiGRU/BiLSTM input via additive injection + LayerNorm on the recycled
    contribution. Raw ESM embeddings are kept as cross-attn K/V and residual on
    every iteration.

    Recycling flow (per iteration):
        recycled  = recycle_input_norm( recycle_proj(prev_enriched) )
        bilstm_in = sequence_embeddings + recycled        # raw ESM + normalised recycle
        local     = BiGRU/BiLSTM(bilstm_in)
        cross_out = cross_attn(Q=local, K/V=sequence_embeddings)   # K/V always raw ESM
        enriched  = LN(cross_out + sequence_embeddings + local)
        prev_enriched = enriched.detach()  if not final iter  (stop-gradient)
                      = enriched           if final iter       (full backprop)

    Zero-init of recycle_proj.weight ensures the first forward pass is identical
    to the base cascDP_Phase1 model

    Training-time stochastic recycling (AF2 pattern):
        n_iters ~ Uniform[1, 1 + num_recycles]  per batch
    Eval-time: always 1 + num_recycles (deterministic, maximum refinement).
    """

    def __init__(
        self,
        backbone,
        device: str = "cuda",
        context_type: str = "bigru",
        dropout: float = 0.5,
        use_crf: bool = False,
        freeze_backbone: bool = False,
        disorder_prior: float = 0.1159,
        num_recycles: int = 2, # number of recycle passes on top of base pass
    ):
        super().__init__(
            backbone=backbone,
            device=device,
            context_type=context_type,
            dropout=dropout,
            use_crf=use_crf,
            freeze_backbone=freeze_backbone,
            disorder_prior=disorder_prior,
        )

        self.num_recycles = num_recycles
        self.recycle_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.recycle_input_norm = nn.LayerNorm(self.hidden_dim)

        # Zero-init so the first iteration is identical to base model
        nn.init.zeros_(self.recycle_proj.weight)

        self.to(device)
        logger.info(
            f"Phase 1 Recycle model created: num_recycles={num_recycles}, "
            f"total passes per forward = 1+{num_recycles} (eval) / stochastic 1..{1+num_recycles} (train)"
        )

    def _run_disorder_pipeline(self, sequence_embeddings: torch.Tensor, return_attention: bool = False):
        # Stochastic depth: sample number of iterations during training
        if self.training:
            n_iters = random.randint(1, 1 + self.num_recycles)
        else:
            n_iters = 1 + self.num_recycles

        # Initialise recycled state to zeros (zero-init; first pass == base model)
        prev_enriched = torch.zeros_like(sequence_embeddings)

        esm_cross_attn_weights = None  # only populated on final iter if requested

        for i in range(n_iters):
            is_final = (i == n_iters - 1)

            # Inject recycled state into BiGRU/BiLSTM input 
            recycled_contribution = self.recycle_input_norm(
                self.recycle_proj(prev_enriched)
            )
            bilstm_in = sequence_embeddings + recycled_contribution

            local_features = self.local_context(bilstm_in)

            # Cross-attention: Q=BiGRU, K/V=raw ESM
            if return_attention and is_final:
                cross_out, esm_cross_attn_weights = self.esm_cross_attn(
                    sequence_embeddings=local_features,
                    disorder_features=sequence_embeddings,
                    need_weights=True,
                )
            else:
                cross_out = self.esm_cross_attn(
                    sequence_embeddings=local_features,
                    disorder_features=sequence_embeddings,
                )

            # Residual fusion (raw ESM anchor preserved)
            enriched = self.esm_cross_attn_norm(
                cross_out + sequence_embeddings + local_features
            )

            # Update recycle state
            # Stop-gradient on all but final iteration so activation memory
            # stays bounded to a single forward graph regardless of n_iters.
            if is_final:
                prev_enriched = enriched          # full backprop on last iter
            else:
                prev_enriched = enriched.detach() # stop-gradient on intermediate iters

        # Post-loop: ASPP + MLP + logits (identical to base class)
        aspp_features = self.disorder_head_stage1(enriched)
        disorder_features = self.disorder_mlp(aspp_features)
        disorder_logits = self.disorder_initial(disorder_features)

        if return_attention:
            return aspp_features, disorder_logits, esm_cross_attn_weights
        return aspp_features, disorder_logits