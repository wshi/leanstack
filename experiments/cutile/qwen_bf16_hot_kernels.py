from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import cuda.tile as ct
import torch

from leanstack.hot_kernels import HotKernelCase, get_hot_kernel_case, list_hot_kernel_cases


_GEMM_KERNELS: dict[tuple[int, int, int], Any] = {}


def _get_bf16_gemm_kernel(m_tile: int, n_tile: int, k_tile: int):
    key = (m_tile, n_tile, k_tile)
    if key in _GEMM_KERNELS:
        return _GEMM_KERNELS[key]

    @ct.kernel
    def _kernel(lhs, rhs, out, k_tiles: ct.Constant[int]):
        pid_m = ct.bid(0)
        pid_n = ct.bid(1)
        acc = ct.zeros((m_tile, n_tile), dtype=ct.float32)
        for k_block in range(k_tiles):
            lhs_tile = ct.load(lhs, index=(pid_m, k_block), shape=(m_tile, k_tile))
            rhs_tile = ct.load(rhs, index=(k_block, pid_n), shape=(k_tile, n_tile))
            acc = ct.mma(lhs_tile, rhs_tile, acc)
        ct.store(out, index=(pid_m, pid_n), tile=ct.astype(acc, ct.bfloat16))

    _GEMM_KERNELS[key] = _kernel
    return _kernel


@ct.kernel
def bf16_rmsnorm_kernel(hidden_states, weight, out, hidden_size: ct.Constant[int], eps: ct.Constant[float]):
    pid_row = ct.bid(0)
    x = ct.load(hidden_states, index=(pid_row, 0), shape=(1, hidden_size))
    w = ct.load(weight, index=(0,), shape=(hidden_size,))
    x_fp32 = ct.astype(x, ct.float32)
    w_fp32 = ct.reshape(ct.astype(w, ct.float32), (1, hidden_size))
    mean_square = ct.sum(x_fp32 * x_fp32, axis=1, keepdims=True) / hidden_size
    inv_rms = ct.rsqrt(mean_square + eps)
    y = ct.astype(x_fp32 * inv_rms * w_fp32, ct.bfloat16)
    ct.store(out, index=(pid_row, 0), tile=y)


def _torch_reference_rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = hidden_states.to(torch.float32)
    w_fp32 = weight.to(torch.float32)
    mean_square = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
    result = x_fp32 * torch.rsqrt(mean_square + eps) * w_fp32
    return result.to(torch.bfloat16)


def _measure_ms(fn, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def _tflops(case: HotKernelCase, median_ms: float) -> float | None:
    flops = case.estimated_flops()
    if flops is None or median_ms <= 0:
        return None
    return float(flops) / (median_ms / 1000.0) / 1e12


def _prepare_gemm_inputs(case: HotKernelCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert case.n is not None
    assert case.k is not None
    lhs = (torch.rand((case.m, case.k), device="cuda", dtype=torch.float32) - 0.5).to(torch.bfloat16)
    rhs = (torch.rand((case.k, case.n), device="cuda", dtype=torch.float32) - 0.5).to(torch.bfloat16)
    out = torch.zeros((case.m, case.n), device="cuda", dtype=torch.bfloat16)
    return lhs, rhs, out


def _prepare_rmsnorm_inputs(case: HotKernelCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert case.hidden_size is not None
    hidden_states = (torch.rand((case.m, case.hidden_size), device="cuda", dtype=torch.float32) - 0.5).to(
        torch.bfloat16
    )
    weight = (torch.rand((case.hidden_size,), device="cuda", dtype=torch.float32) - 0.5).to(torch.bfloat16)
    out = torch.zeros_like(hidden_states)
    return hidden_states, weight, out


def _run_gemm_case(case: HotKernelCase, warmup: int, repeats: int) -> dict[str, Any]:
    assert case.n is not None
    assert case.k is not None
    assert case.m_tile is not None
    assert case.n_tile is not None
    assert case.k_tile is not None
    if case.m % case.m_tile != 0:
        raise ValueError(f"{case.key} requires m divisible by m_tile ({case.m} vs {case.m_tile})")
    if case.n % case.n_tile != 0:
        raise ValueError(f"{case.key} requires n divisible by n_tile ({case.n} vs {case.n_tile})")
    if case.k % case.k_tile != 0:
        raise ValueError(f"{case.key} requires k divisible by k_tile ({case.k} vs {case.k_tile})")

    lhs, rhs, out = _prepare_gemm_inputs(case)
    expected = torch.matmul(lhs, rhs)
    stream = torch.cuda.current_stream().cuda_stream
    grid = (case.m // case.m_tile, case.n // case.n_tile, 1)
    k_tiles = case.k // case.k_tile
    kernel = _get_bf16_gemm_kernel(case.m_tile, case.n_tile, case.k_tile)

    def _run_cutile() -> None:
        ct.launch(stream, grid, kernel, (lhs, rhs, out, k_tiles))

    def _run_torch() -> None:
        torch.matmul(lhs, rhs)

    cutile_timing = _measure_ms(_run_cutile, warmup=warmup, repeats=repeats)
    torch_timing = _measure_ms(_run_torch, warmup=warmup, repeats=repeats)
    observed = out.to(torch.float32)
    reference = expected.to(torch.float32)

    return {
        "case": case.as_payload(),
        "grid": grid,
        "k_tiles": k_tiles,
        "correctness": {
            "max_abs_error": float(torch.max(torch.abs(observed - reference)).item()),
            "mean_abs_error": float(torch.mean(torch.abs(observed - reference)).item()),
        },
        "timings_ms": {
            "cutile": cutile_timing,
            "torch": torch_timing,
        },
        "throughput": {
            "cutile_tflops": _tflops(case, cutile_timing["median_ms"]),
            "torch_tflops": _tflops(case, torch_timing["median_ms"]),
            "speedup_vs_torch": (
                torch_timing["median_ms"] / cutile_timing["median_ms"]
                if cutile_timing["median_ms"] > 0
                else None
            ),
        },
    }


def _run_rmsnorm_case(case: HotKernelCase, warmup: int, repeats: int) -> dict[str, Any]:
    assert case.hidden_size is not None
    assert case.eps is not None
    hidden_states, weight, out = _prepare_rmsnorm_inputs(case)
    expected = _torch_reference_rmsnorm(hidden_states, weight, case.eps)
    stream = torch.cuda.current_stream().cuda_stream
    grid = (case.m, 1, 1)

    def _run_cutile() -> None:
        ct.launch(stream, grid, bf16_rmsnorm_kernel, (hidden_states, weight, out, case.hidden_size, case.eps))

    def _run_torch() -> None:
        _torch_reference_rmsnorm(hidden_states, weight, case.eps)

    cutile_timing = _measure_ms(_run_cutile, warmup=warmup, repeats=repeats)
    torch_timing = _measure_ms(_run_torch, warmup=warmup, repeats=repeats)
    observed = out.to(torch.float32)
    reference = expected.to(torch.float32)

    return {
        "case": case.as_payload(),
        "grid": grid,
        "correctness": {
            "max_abs_error": float(torch.max(torch.abs(observed - reference)).item()),
            "mean_abs_error": float(torch.mean(torch.abs(observed - reference)).item()),
        },
        "timings_ms": {
            "cutile": cutile_timing,
            "torch": torch_timing,
        },
        "throughput": {
            "cutile_tflops": _tflops(case, cutile_timing["median_ms"]),
            "torch_tflops": _tflops(case, torch_timing["median_ms"]),
            "speedup_vs_torch": (
                torch_timing["median_ms"] / cutile_timing["median_ms"]
                if cutile_timing["median_ms"] > 0
                else None
            ),
        },
    }


def _resolve_cases(keys: list[str] | None) -> list[HotKernelCase]:
    if keys:
        return [get_hot_kernel_case(key) for key in keys]
    return list(list_hot_kernel_cases(default_only=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BF16 Qwen3-4B hot kernels through cuTile.")
    parser.add_argument("--cases", nargs="*", help="Specific hot-kernel case keys to run.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cases = _resolve_cases(args.cases)
    results = []
    for case in cases:
        if case.kernel_kind == "gemm":
            results.append(_run_gemm_case(case, warmup=args.warmup, repeats=args.repeats))
        elif case.kernel_kind == "rmsnorm":
            results.append(_run_rmsnorm_case(case, warmup=args.warmup, repeats=args.repeats))
        else:
            raise ValueError(f"unsupported kernel kind: {case.kernel_kind}")

    payload = {
        "suite": "qwen3_4b_bf16_hot_kernels",
        "model_id": cases[0].model_id if cases else "unknown",
        "dtype": cases[0].dtype if cases else "unknown",
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        "compiler_artifacts": {
            "bytecode_dir": os.environ.get("CUDA_TILE_DUMP_BYTECODE"),
            "tileir_dir": os.environ.get("CUDA_TILE_DUMP_TILEIR"),
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
