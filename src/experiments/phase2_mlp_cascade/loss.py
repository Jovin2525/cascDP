"""
src/experiments/phase2_mlp_cascade/loss.py

Re-exports Phase2Loss for use in MLP-cascade experiment training scripts.
Loss function is identical to standard Phase 2 — no disorder term (Phase 1 is frozen).
"""

from ...training.loss import Phase2Loss as MLPCascadeLoss  # noqa: F401
