"""OpenAI-compatible /v1/embeddings server for EmbeddingGemma-2 on MLX.

    python -m embeddinggemma_mlx.server --model ./embeddinggemma-2-mlx --port 7341
    EMBEDDINGGEMMA_MOCK=1 python -m embeddinggemma_mlx.server   # CI / no weights

Endpoints: POST /v1/embeddings, GET /v1/models, GET /healthz.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .matryoshka import SUPPORTED_DIMS
from .model import MockEmbeddingGemma, MlxEmbeddingGemma

_SUPPORTED_DIMS_LIST = list(SUPPORTED_DIMS)


class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, list[str]]
    dims: int | None = None


def _build_app(model_path: str, mock: bool) -> Any:
    app = FastAPI(title="embeddinggemma-mlx", version="0.1.0")
    embedder = MockEmbeddingGemma(model_path) if mock else MlxEmbeddingGemma(model_path)
    model_name = os.path.basename(model_path.rstrip("/")) or model_path

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "served_by": "mock" if mock else "mlx_lm", "model": model_name}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "embeddinggemma-mlx"}],
        }

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest) -> JSONResponse:
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise HTTPException(status_code=400, detail="input must be a non-empty string or list of non-empty strings")
        if req.model != model_name:
            raise HTTPException(status_code=404, detail=f"unknown model {req.model!r}; served: {model_name!r}")
        if req.dims is not None and req.dims not in _SUPPORTED_DIMS_LIST:
            raise HTTPException(status_code=400, detail=f"dims must be one of {_SUPPORTED_DIMS_LIST}")
        data = []
        total = 0
        started = time.perf_counter()
        for i, text in enumerate(texts):
            result = embedder.embed(text, dims=req.dims)
            data.append({"object": "embedding", "index": i, "embedding": result.embedding.tolist()})
            total += result.prompt_tokens
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": model_name,
                "usage": {"prompt_tokens": total, "total_tokens": total},
                "served_by": "mock" if mock else "mlx_lm",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="EmbeddingGemma-2 MLX embeddings server")
    parser.add_argument("--model", default=os.environ.get("EMBEDDINGGEMMA_MODEL", "google/embeddinggemma-2"))
    parser.add_argument("--host", default=os.environ.get("EMBEDDINGGEMMA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EMBEDDINGGEMMA_PORT", "7341")))
    parser.add_argument("--mock", action="store_true", default=os.environ.get("EMBEDDINGGEMMA_MOCK", "") in ("1", "true", "yes"))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(_build_app(args.model, args.mock), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
