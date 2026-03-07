from __future__ import annotations

import argparse
import json
from pathlib import Path

import cuda.tile as ct
import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a torch-backed cuTile vector add kernel.")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="float16")
    parser.add_argument("--vector-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--input-modulus", type=int, default=64)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vector_size % args.tile_size != 0:
        raise SystemExit("--vector-size must be divisible by --tile-size")
    if args.input_modulus <= 0:
        raise SystemExit("--input-modulus must be positive")

    dtype = DTYPE_MAP[args.dtype]
    # Keep the numerical range bounded so the gate measures compiler reachability
    # instead of large-magnitude BF16 rounding behavior.
    base = torch.remainder(
        torch.arange(args.vector_size, device="cuda", dtype=torch.float32),
        float(args.input_modulus),
    )
    a = base.to(dtype)
    b = (base * 2).to(dtype)
    c = torch.zeros(args.vector_size, device="cuda", dtype=dtype)

    grid = (args.vector_size // args.tile_size, 1, 1)
    stream = torch.cuda.current_stream().cuda_stream
    ct.launch(stream, grid, vector_add, (a, b, c, args.tile_size))
    torch.cuda.synchronize()

    expected = (a.to(torch.float32) + b.to(torch.float32)).to(torch.float32)
    observed = c.to(torch.float32)
    max_error = float(torch.max(torch.abs(observed - expected)).item())
    result = {
        "dtype": args.dtype,
        "vector_size": args.vector_size,
        "tile_size": args.tile_size,
        "input_modulus": args.input_modulus,
        "grid": grid,
        "max_error": max_error,
        "sample": observed[:8].cpu().tolist(),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
