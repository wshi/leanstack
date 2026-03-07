from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from leanstack.runtime.qwen_explicit import (
    materialize_qwen_block_runtime,
    materialize_qwen_semantic_block_runtime,
    run_block_forward,
    run_semantic_block_forward,
    run_semantic_single_layer_decode,
    run_semantic_single_layer_prefill,
    run_single_layer_decode,
    run_single_layer_prefill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare borrowed and semantic Qwen block implementations.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prompt", default="Explain why fixing the model-chip contract simplifies inference software.")
    parser.add_argument("--prompt-format", choices=("auto", "chat", "raw"), default="chat")
    parser.add_argument("--max-prefill-tokens", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_thinking_mode(args: argparse.Namespace) -> bool | None:
    if args.enable_thinking:
        return True
    if args.disable_thinking:
        return False
    return None


def build_prompt(tokenizer, args: argparse.Namespace, thinking_mode: bool | None) -> tuple[str, str]:
    if args.prompt_format == "raw":
        return args.prompt, "raw"

    use_chat = args.prompt_format == "chat" or getattr(tokenizer, "chat_template", None) is not None
    if not use_chat:
        return args.prompt, "raw"

    messages = [{"role": "user", "content": args.prompt}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking_mode is not None:
        template_kwargs["enable_thinking"] = thinking_mode
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs), "chat"
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **template_kwargs), "chat"


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference - candidate).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    thinking_mode = resolve_thinking_mode(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    formatted_prompt, resolved_prompt_format = build_prompt(tokenizer, args, thinking_mode)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][:, : args.max_prefill_tokens]
    decode_ids = input_ids[:, -1:].clone()

    timings: dict[str, float] = {}

    sync_device(device)
    start = time.perf_counter()
    borrowed_runtime = materialize_qwen_block_runtime(
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        device=args.device,
        dtype=args.dtype,
    )
    sync_device(device)
    timings["borrowed_materialize_seconds"] = time.perf_counter() - start

    sync_device(device)
    start = time.perf_counter()
    semantic_runtime = materialize_qwen_semantic_block_runtime(
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        device=args.device,
        dtype=args.dtype,
    )
    sync_device(device)
    timings["semantic_materialize_seconds"] = time.perf_counter() - start

    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        borrowed_forward = run_block_forward(borrowed_runtime, input_ids)
        sync_device(device)
        timings["borrowed_forward_seconds"] = time.perf_counter() - start

        sync_device(device)
        start = time.perf_counter()
        semantic_forward = run_semantic_block_forward(semantic_runtime, input_ids)
        sync_device(device)
        timings["semantic_forward_seconds"] = time.perf_counter() - start

        sync_device(device)
        start = time.perf_counter()
        borrowed_prefill, borrowed_cache = run_single_layer_prefill(borrowed_runtime, input_ids)
        sync_device(device)
        timings["borrowed_prefill_seconds"] = time.perf_counter() - start

        sync_device(device)
        start = time.perf_counter()
        semantic_prefill, semantic_cache = run_semantic_single_layer_prefill(
            semantic_runtime,
            input_ids,
            page_size=args.page_size,
            max_seq_len=int(input_ids.shape[-1]) + 1,
        )
        sync_device(device)
        timings["semantic_prefill_seconds"] = time.perf_counter() - start

        sync_device(device)
        start = time.perf_counter()
        borrowed_decode, borrowed_cache = run_single_layer_decode(borrowed_runtime, decode_ids, borrowed_cache)
        sync_device(device)
        timings["borrowed_decode_seconds"] = time.perf_counter() - start

        sync_device(device)
        start = time.perf_counter()
        semantic_decode, semantic_cache = run_semantic_single_layer_decode(
            semantic_runtime,
            decode_ids,
            semantic_cache,
        )
        sync_device(device)
        timings["semantic_decode_seconds"] = time.perf_counter() - start

    result = {
        "model_path": args.model_path,
        "layer_idx": args.layer_idx,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_format": resolved_prompt_format,
        "thinking_mode": (
            "enabled" if thinking_mode is True else "disabled" if thinking_mode is False else "default"
        ),
        "prefill_tokens": int(input_ids.shape[-1]),
        "decode_tokens": int(decode_ids.shape[-1]),
        "borrowed_cache_seq_length": int(borrowed_cache.get_seq_length()),
        "semantic_cache_seq_length": int(semantic_cache.get_seq_length()),
        "semantic_cache_summary": semantic_cache.summary(),
        "forward_diff": diff_stats(borrowed_forward, semantic_forward),
        "prefill_diff": diff_stats(borrowed_prefill, semantic_prefill),
        "decode_diff": diff_stats(borrowed_decode, semantic_decode),
        "timings": timings,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
