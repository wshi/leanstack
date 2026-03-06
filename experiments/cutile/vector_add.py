from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cupy as cp
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a known-good cuTile vector add kernel.")
    parser.add_argument("--vector-size", type=int, default=1 << 20)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--artifact-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vector_size % args.tile_size != 0:
        raise SystemExit("--vector-size must be divisible by --tile-size for this smoke kernel")

    a = cp.arange(args.vector_size, dtype=cp.float32)
    b = cp.arange(args.vector_size, dtype=cp.float32) * 2
    c = cp.zeros_like(a)
    grid = (ct.cdiv(args.vector_size, args.tile_size), 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, c, args.tile_size))
    cp.cuda.get_current_stream().synchronize()

    expected = a + b
    max_error = float(cp.max(cp.abs(c - expected)).item())
    result = {
        "vector_size": args.vector_size,
        "tile_size": args.tile_size,
        "grid": grid,
        "max_error": max_error,
        "bytecode_dir": os.environ.get("CUDA_TILE_DUMP_BYTECODE"),
        "tileir_dir": os.environ.get("CUDA_TILE_DUMP_TILEIR"),
    }

    if not cp.allclose(c, expected):
        raise SystemExit(json.dumps(result, indent=2))

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        (args.artifact_dir / "vector_add_result.json").write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

