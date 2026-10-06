"""embeddinggemma-mlx: EmbeddingGemma 2 text-tower embeddings on Apple Silicon MLX.

Public surface:
    from embeddinggemma_mlx import MlxEmbeddingGemma, MockEmbeddingGemma, truncate_normalize

See README for conversion, serving, and the Matryoshka dims contract.
"""

from .matryoshka import SUPPORTED_DIMS, l2_normalize, truncate_normalize
from .model import EmbeddingResult, MockEmbeddingGemma, MlxEmbeddingGemma
from .pooling import last_token_pool, mean_pool

__version__ = "0.1.0"
__all__ = [
    "SUPPORTED_DIMS",
    "EmbeddingResult",
    "MockEmbeddingGemma",
    "MlxEmbeddingGemma",
    "l2_normalize",
    "truncate_normalize",
    "last_token_pool",
    "mean_pool",
]
