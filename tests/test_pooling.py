import numpy as np
import pytest

from embeddinggemma_mlx.pooling import last_token_pool, mean_pool


def test_last_token_pool_takes_final_position():
    hidden = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.allclose(last_token_pool(hidden), hidden[-1])


def test_last_token_pool_rejects_empty():
    with pytest.raises(ValueError):
        last_token_pool(np.zeros((0, 3), dtype=np.float32))


def test_mean_pool_respects_pad_mask():
    hidden = np.array([[1.0, 0.0], [2.0, 0.0], [99.0, 0.0]], dtype=np.float32)
    mask = np.array([1.0, 1.0, 0.0])
    out = mean_pool(hidden, pad_mask=mask)
    assert np.allclose(out, [1.5, 0.0])


def test_mean_pool_all_masked_rejected():
    with pytest.raises(ValueError):
        mean_pool(np.ones((3, 2), dtype=np.float32), pad_mask=np.zeros(3))


def test_eos_shifts_last_token():
    """Appending EOS must change the pooled vector (it is a new last position)."""
    rng = np.random.default_rng(5)
    base = rng.standard_normal((5, 8)).astype(np.float32)
    eos = rng.standard_normal((1, 8)).astype(np.float32)
    without = last_token_pool(base)
    with_eos = last_token_pool(np.vstack([base, eos]))
    assert not np.allclose(without, with_eos)
