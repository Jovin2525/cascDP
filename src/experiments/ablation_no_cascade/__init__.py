"""
src/experiments/ablation_no_cascade — No-Disorder-Cascade Ablation

Tests whether the Phase 1 disorder-cascade improves binding / linker prediction.

Architecture difference vs cascDP_Phase2:
    Full Phase 2:   ESM -> [frozen Phase1 disorder pipeline] -> function heads
    Ablation:      ESM -> BiGRU -> function heads  (disorder pipeline bypassed)

Phase 1 is NOT instantiated.  Instead the Phase 1 checkpoint supplies 
backbone weights which are loaded directly into the backbone.
"""

from .model import cascDP_Ablation1
from .loss import AblationLoss
from .trainer import AblationTrainer

__all__ = ["cascDP_Ablation1", "AblationLoss", "AblationTrainer"]
