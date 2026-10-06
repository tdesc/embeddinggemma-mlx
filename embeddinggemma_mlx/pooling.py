"""Pooling for causal-LM embedding models.

EmbeddingGemma follows the last-token recipe: append the EOS token, then take
the hidden state of the final position. Texts are embedded one at a time (no
padding), so "last position" is unambiguous. Pure NumPy / framework-agnostic
so it is unit-testable without weights.
"""

from __future__ import annotations

import numpy as np


def last_token_pool(hidden: np.ndarray) -> np.ndarray:
    """Pool the last position of a (seq, dim) hidden-state matrix."""
    if hidden.ndim != 2:
        raise ValueError(f"expected (seq, dim) matrix, got shape {hidden.shape}")
    if hidden.shape[0] == 0:
        raise ValueError("empty sequence")
    return hidden[-1, :]


def mean_pool(hidden: np.ndarray, pad_mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over non-pad positions; ``pad_mask`` is 1 for real tokens."""
    if hidden.ndim != 2:
        raise ValueError(f"expected (seq, dim) matrix, got shape {hidden.shape}")
    if pad_mask is None:
        return hidden.mean(axis=0)
    mask = np.asarray(pad_mask, dtype=np.float32)
    if mask.shape[0] != hidden.shape[0]:
        raise ValueError("pad_mask length must equal sequence length")
    denom = mask.sum()
    if denom <= 0:
        raise ValueError("all positions masked")
    return (hidden * mask[:, None]).sum(axis=0) / denom
