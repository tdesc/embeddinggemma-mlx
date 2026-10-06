"""Matryoshka embedding truncation: slice leading dims, then re-normalize.

Pure NumPy so it runs (and is tested) on any CI runner without MLX.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_DIMS = (768, 512, 256, 128)


def truncate_normalize(vec: np.ndarray, dims: int) -> np.ndarray:
    """Return the first ``dims`` components of ``vec``, L2-renormalized."""
    if dims not in SUPPORTED_DIMS:
        raise ValueError(f"dims must be one of {SUPPORTED_DIMS}, got {dims}")
    if vec.ndim != 1:
        raise ValueError(f"expected 1-D vector, got shape {vec.shape}")
    if vec.shape[0] < dims:
        raise ValueError(f"vector dim {vec.shape[0]} < requested {dims}")
    out = np.asarray(vec[:dims], dtype=np.float32)
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (out / norm).astype(np.float32)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (out / norm).astype(np.float32)
