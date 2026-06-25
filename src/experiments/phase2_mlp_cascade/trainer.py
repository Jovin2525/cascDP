"""
src/experiments/phase2_mlp_cascade/trainer.py

Re-exports the main Trainer for use in MLP-cascade training scripts.
cascDP_Phase2_MLPCascade is forward-compatible with the standard Trainer
(same output signature: disorder_logits, binding_logits, linker_logits).
"""

from ...training.trainer import Trainer as MLPCascadeTrainer  # noqa: F401
