"""
Embedding generation utilities.
"""

__all__ = ["EmbeddingGenerator"]


def __getattr__(name):
    if name == "EmbeddingGenerator":
        from .generate_embeddings import EmbeddingGenerator

        return EmbeddingGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
