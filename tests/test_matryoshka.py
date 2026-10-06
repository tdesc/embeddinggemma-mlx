import numpy as np
import pytest

from embeddinggemma_mlx.matryoshka import SUPPORTED_DIMS, l2_normalize, truncate_normalize


def _unit(d: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal(d).astype(np.float32))


def test_supported_dims_match_model_card():
    assert SUPPORTED_DIMS == (768, 512, 256, 128)


def test_truncate_keeps_prefix_and_unit_norm():
    v = _unit(768, seed=3)
    for dims in SUPPORTED_DIMS:
        out = truncate_normalize(v, dims)
        assert out.shape == (dims,)
        assert np.allclose(out, v[:dims] / np.linalg.norm(v[:dims]), atol=1e-6)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5


def test_truncate_prefix_of_smaller_matryoshka_matches_direct():
    v = _unit(768, seed=11)
    direct256 = truncate_normalize(v, 256)
    via512 = truncate_normalize(truncate_normalize(v, 512), 256)
    assert np.allclose(direct256, via512, atol=1e-5), "nested truncation must be consistent"


def test_unsupported_dims_rejected():
    v = _unit(768)
    with pytest.raises(ValueError):
        truncate_normalize(v, 384)


def test_zero_vector_rejected():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(768, dtype=np.float32))
