import numpy as np
from fastapi.testclient import TestClient

from embeddinggemma_mlx.server import _build_app


def make_client():
    app = _build_app("embeddinggemma-2-mlx", mock=True)
    return TestClient(app), "embeddinggemma-2-mlx"


def test_healthz_and_models():
    client, name = make_client()
    assert client.get("/healthz").json()["served_by"] == "mock"
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == name


def test_embeddings_openai_shape_single_and_batch():
    client, name = make_client()
    r = client.post("/v1/embeddings", json={"model": name, "input": "привет мир"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 768
    assert body["usage"]["total_tokens"] > 0

    r2 = client.post("/v1/embeddings", json={"model": name, "input": ["a", "b c"]})
    assert r2.status_code == 200
    assert [d["index"] for d in r2.json()["data"]] == [0, 1]


def test_embeddings_dims_matryoshka_and_unit_norm():
    client, name = make_client()
    for dims in (768, 512, 256, 128):
        r = client.post("/v1/embeddings", json={"model": name, "input": "x", "dims": dims})
        assert r.status_code == 200
        vec = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        assert vec.shape == (dims,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_mock_is_deterministic_and_prefix_consistent():
    client, name = make_client()
    body = lambda **kw: client.post("/v1/embeddings", json={"model": name, "input": "одна и та же строка", **kw}).json()
    v768 = np.array(body()["data"][0]["embedding"])
    v768_again = np.array(body()["data"][0]["embedding"])
    assert np.allclose(v768, v768_again)
    v256 = np.array(body(dims=256)["data"][0]["embedding"])
    assert np.allclose(v256, v768[:256] / np.linalg.norm(v768[:256]), atol=1e-5)


def test_negative_controls():
    client, name = make_client()
    assert client.post("/v1/embeddings", json={"model": name, "input": ""}).status_code == 400
    assert client.post("/v1/embeddings", json={"model": "other", "input": "x"}).status_code == 404
    r = client.post("/v1/embeddings", json={"model": name, "input": "x", "dims": 384})
    assert r.status_code == 400
