"""EmbeddingGemma-2 text tower on MLX.

Loads the (converted) model with ``mlx_lm``, appends EOS, captures the last
transformer layer's hidden state via a forward hook (the same technique as
Colony's hidden-states gateway), applies last-token pooling, L2-normalizes,
and optionally truncates to Matryoshka dims.

All MLX forward passes are serialized on one dedicated thread: Metal GPU
streams are thread-local and a cross-thread call raises
``std::runtime_error: There is no Stream(gpu, 1) in current thread``.

Quantization is intentionally NOT applied by default: embedding quality is
sensitive to weight compression. Convert in fp16.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import numpy as np

from .matryoshka import SUPPORTED_DIMS, l2_normalize, truncate_normalize
from .pooling import last_token_pool

T = TypeVar("T")


class _MlxDispatchExecutor:
    """Process-wide single MLX worker thread (see module docstring)."""

    def __init__(self) -> None:
        import queue

        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="embeddinggemma-mlx-dispatch", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import queue as _q

        while True:
            fn, args, reply = self._queue.get()
            try:
                reply.put((True, fn(*args)))
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                reply.put((False, exc))
            finally:
                self._queue.task_done()

    def run(self, fn: Callable[..., T]) -> T:
        import queue as _q

        reply: "queue.Queue[Any]" = _q.Queue()
        self._queue.put((fn, (), reply))
        ok, payload = reply.get()
        if not ok:
            raise payload
        return payload


@dataclass
class EmbeddingResult:
    embedding: np.ndarray
    prompt_tokens: int


class MlxEmbeddingGemma:
    """Last-token embedding of the EmbeddingGemma-2 text tower."""

    def __init__(self, model_path: str, *, dims: Optional[int] = None):
        if dims is not None and dims not in SUPPORTED_DIMS:
            raise ValueError(f"dims must be one of {SUPPORTED_DIMS}, got {dims}")
        self.model_path = model_path
        self.dims = dims
        self._model: Any = None
        self._tokenizer: Any = None
        self._capture: list[np.ndarray] = []
        self._executor = _MlxDispatchExecutor()

    # -- lazy load ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        def _load() -> None:
            import mlx.core as mx

            import mlx_lm

            model, tokenizer = mlx_lm.load(self.model_path)
            inner = getattr(model, "model", model)
            layers = getattr(inner, "layers", None)
            if not layers:
                raise RuntimeError(
                    f"model {type(inner).__name__} does not expose .layers; "
                    "last-hidden capture is not implemented for this architecture"
                )
            last_layer = layers[-1]
            original = last_layer.__call__
            capture = self._capture

            def wrapped(x: Any, *args: Any, **kwargs: Any) -> Any:
                out = original(x, *args, **kwargs)
                try:
                    arr = out
                    # mlx_lm layers may return (hidden,) tuples or hidden directly
                    if isinstance(arr, tuple):
                        arr = arr[0]
                    capture.append(np.asarray(arr, dtype=np.float32))
                except Exception:  # noqa: BLE001 - capture must never break forward
                    pass
                return out

            last_layer.__call__ = wrapped
            self._model = model
            self._tokenizer = tokenizer

        self._executor.run(_load)

    # -- embedding ---------------------------------------------------------

    def embed(self, text: str, *, dims: Optional[int] = None) -> EmbeddingResult:
        dims = dims if dims is not None else self.dims
        if dims is not None and dims not in SUPPORTED_DIMS:
            raise ValueError(f"dims must be one of {SUPPORTED_DIMS}, got {dims}")
        self._ensure_loaded()

        def _forward() -> np.ndarray:
            import mlx.core as mx

            tokenizer = self._tokenizer
            eos = getattr(getattr(tokenizer, "eos_token", None), "token", None) or getattr(
                tokenizer, "eos_token", None
            )
            ids = tokenizer.encode(text)
            if eos is not None:
                eos_id = getattr(eos, "id", None)
                if eos_id is None:
                    try:
                        eos_id = tokenizer.convert_tokens_to_ids(eos)
                    except Exception:  # noqa: BLE001 - model without eos in vocab
                        eos_id = None
                if eos_id is not None:
                    ids = list(ids) + [eos_id]
            if not ids:
                raise ValueError("empty token sequence")
            tokens = mx.array([ids])
            self._capture.clear()
            # One forward pass; generation-free: model(input_ids) evaluates layers.
            _ = self._model(tokens)
            if not self._capture:
                raise RuntimeError("last-layer capture did not fire; check hook install")
            hidden = self._capture[-1]
            return hidden

        hidden = self._executor.run(_forward)
        vec = l2_normalize(last_token_pool(hidden))
        if dims is not None:
            vec = truncate_normalize(vec, dims)
        return EmbeddingResult(embedding=vec, prompt_tokens=len(ids))  # noqa: F821 - ids set in closure


class MockEmbeddingGemma:
    """Deterministic hash-based embedder with the same interface (CI, no weights)."""

    FULL_DIM = 768

    def __init__(self, model_path: str = "mock", *, dims: Optional[int] = None):
        if dims is not None and dims not in SUPPORTED_DIMS:
            raise ValueError(f"dims must be one of {SUPPORTED_DIMS}, got {dims}")
        self.model_path = model_path
        self.dims = dims

    def embed(self, text: str, *, dims: Optional[int] = None) -> EmbeddingResult:
        import hashlib

        dims = dims if dims is not None else self.dims
        if dims is not None and dims not in SUPPORTED_DIMS:
            raise ValueError(f"dims must be one of {SUPPORTED_DIMS}, got {dims}")
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(seed * ((self.FULL_DIM // len(seed)) + 1), dtype=np.uint8)[: self.FULL_DIM]
        vec = l2_normalize(raw.astype(np.float32) - 128.0)
        if dims is not None:
            vec = truncate_normalize(vec, dims)
        return EmbeddingResult(embedding=vec, prompt_tokens=max(1, len(text.split())))
