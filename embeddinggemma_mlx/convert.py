"""CLI: convert google/embeddinggemma-2 weights to MLX (fp16 by default).

Usage:
    python -m embeddinggemma_mlx.convert --hf-path google/embeddinggemma-2 \
        --mlx-path ./embeddinggemma-2-mlx

Embedding models are sensitive to weight compression: pass --quantize only if
you accept the quality risk and have a canary to verify it.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert EmbeddingGemma-2 to MLX")
    parser.add_argument("--hf-path", default="google/embeddinggemma-2")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--quantize", action="store_true",
                        help="quantize weights (NOT recommended for embeddings)")
    parser.add_argument("--q-bits", type=int, default=8)
    args = parser.parse_args()

    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise SystemExit("mlx_lm is not installed: pip install mlx_lm") from exc

    if args.quantize:
        print(f"WARNING: quantizing to {args.q_bits}-bit degrades embedding quality; "
              "run a retrieval canary before adopting.")
    else:
        print("Converting in fp16 (no quantization) — recommended for embeddings.")

    import mlx_lm

    mlx_lm.convert(  # callable re-exported by the package (mlx_lm >=0.19)
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_bits=args.q_bits,
    )
    print(f"Done. Serve it: python -m embeddinggemma_mlx.server --model {args.mlx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
