"""
cascDP - Cascaded Disorder & Function Prediction for Intrinsically Disordered Proteins.
"""

__version__ = "0.1.0"

__all__ = [
    'cascDP_Phase1',
    'cascDP_Phase2',
    'create_backbone',
    'ProteinBackbone',
    'ESMCBackbone',
    'EmbeddingGenerator',
    'FUNCTIONAL_CLASS_MAPPING_99',
]


def __getattr__(name):
    if name in {
        'cascDP_Phase1',
        'cascDP_Phase2',
        'create_backbone',
        'ProteinBackbone',
        'ESMCBackbone',
    }:
        from .models import (
            ESMCBackbone,
            ProteinBackbone,
            cascDP_Phase1,
            cascDP_Phase2,
            create_backbone,
        )

        return {
            'cascDP_Phase1': cascDP_Phase1,
            'cascDP_Phase2': cascDP_Phase2,
            'create_backbone': create_backbone,
            'ProteinBackbone': ProteinBackbone,
            'ESMCBackbone': ESMCBackbone,
        }[name]

    if name == 'FUNCTIONAL_CLASS_MAPPING_99':
        from .data.preprocessing import FUNCTIONAL_CLASS_MAPPING_99

        return FUNCTIONAL_CLASS_MAPPING_99

    if name == 'EmbeddingGenerator':
        from .embeddings import EmbeddingGenerator

        return EmbeddingGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
