from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from leanstack.runtime.qwen_explicit import (
    materialize_qwen_full_runtime,
    materialize_qwen_stack_runtime,
    normalize_stop_token_ids,
    project_hidden_to_logits,
    run_stack_decode,
    run_stack_prefill,
    select_greedy_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an explicit Qwen3 runtime loop on GPU.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-layers", type=int, default=0, help="0 means the full model.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prompt", default="Explain why fixing the model-chip contract simplifies inference software.")
    parser.add_argument("--prompt-format", choices=("auto", "chat", "raw"), default="chat")
    parser.add_argument("--max-prefill-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--stop-token-id", action="append", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--skip-final-cache-advance", action="store_true")
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


def snapshot_cuda_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def resolve_stop_token_ids(tokenizer, args: argparse.Namespace) -> tuple[int, ...]:
    stop_ids = set()
    if not args.ignore_eos:
        stop_ids.update(normalize_stop_token_ids(getattr(tokenizer, "eos_token_id", None)))
    if args.stop_token_id:
        stop_ids.update(int(token_id) for token_id in args.stop_token_id)
    return tuple(sorted(stop_ids))


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.num_layers < 0:
        raise ValueError("--num-layers must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    thinking_mode = resolve_thinking_mode(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    formatted_prompt, resolved_prompt_format = build_prompt(tokenizer, args, thinking_mode)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][:, : args.max_prefill_tokens]
    stop_token_ids = resolve_stop_token_ids(tokenizer, args)

    timings: dict[str, float] = {}
    memory: dict[str, dict[str, int] | None] = {"before_materialize": snapshot_cuda_memory(device)}

    sync_device(device)
    start = time.perf_counter()
    if args.num_layers == 0:
        runtime = materialize_qwen_full_runtime(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            include_output_head=True,
        )
    else:
        runtime = materialize_qwen_stack_runtime(
            model_path=args.model_path,
            layer_indices=tuple(range(args.num_layers)),
            device=args.device,
            dtype=args.dtype,
            include_output_head=True,
        )
    sync_device(device)
    timings["materialize_seconds"] = time.perf_counter() - start
    memory["after_materialize"] = snapshot_cuda_memory(device)

    generated_ids: list[int] = []
    decode_step_seconds: list[float] = []
    stop_reason = "max_new_tokens"

    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        prefill_hidden, cache = run_stack_prefill(runtime, input_ids)
        next_logits = project_hidden_to_logits(runtime, prefill_hidden)
        sync_device(device)
        timings["prefill_seconds"] = time.perf_counter() - start

        loop_start = time.perf_counter()
        for step in range(args.max_new_tokens):
            next_token = select_greedy_token(next_logits)
            token_id = int(next_token.item())
            generated_ids.append(token_id)

            if token_id in stop_token_ids:
                stop_reason = "stop_token"
                break

            if step == args.max_new_tokens - 1:
                break

            sync_device(device)
            step_start = time.perf_counter()
            decode_hidden, cache = run_stack_decode(runtime, next_token, cache)
            next_logits = project_hidden_to_logits(runtime, decode_hidden)
            sync_device(device)
            decode_step_seconds.append(time.perf_counter() - step_start)

        sync_device(device)
        timings["decode_loop_seconds"] = time.perf_counter() - loop_start

        timings["final_cache_advance_seconds"] = 0.0
        final_cache_advanced = False
        if generated_ids and not args.skip_final_cache_advance:
            final_token = torch.tensor([[generated_ids[-1]]], device=device, dtype=torch.long)
            sync_device(device)
            start = time.perf_counter()
            _, cache = run_stack_decode(runtime, final_token, cache)
            sync_device(device)
            timings["final_cache_advance_seconds"] = time.perf_counter() - start
            final_cache_advanced = True

    generated_tensor = torch.tensor([generated_ids], dtype=torch.long)
    generated_text = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
    runtime_loop_seconds = timings["prefill_seconds"] + timings["decode_loop_seconds"]
    full_loop_seconds = runtime_loop_seconds + timings["final_cache_advance_seconds"]
    emitted_tokens = len(generated_ids)
    memory["after_runtime_loop"] = snapshot_cuda_memory(device)

    result = {
        "model_path": args.model_path,
        "num_layers": len(runtime.layer_indices),
        "layer_range": [runtime.layer_indices[0], runtime.layer_indices[-1]] if runtime.layer_indices else [],
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": runtime.config._attn_implementation,
        "prompt_format": resolved_prompt_format,
        "thinking_mode": (
            "enabled" if thinking_mode is True else "disabled" if thinking_mode is False else "default"
        ),
        "prompt_tokens": int(input_ids.shape[-1]),
        "max_new_tokens": args.max_new_tokens,
        "emitted_tokens": emitted_tokens,
        "decode_steps_executed": len(decode_step_seconds),
        "stop_reason": stop_reason,
        "stop_token_ids": list(stop_token_ids),
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "cache_seq_length": int(cache.get_seq_length()),
        "final_cache_advanced": final_cache_advanced,
        "timings": {
            **timings,
            "runtime_loop_seconds": runtime_loop_seconds,
            "full_loop_seconds": full_loop_seconds,
            "decode_step_seconds": decode_step_seconds,
        },
        "throughput": {
            "runtime_tokens_per_second": (emitted_tokens / runtime_loop_seconds) if runtime_loop_seconds > 0 else None,
            "full_loop_tokens_per_second": (emitted_tokens / full_loop_seconds) if full_loop_seconds > 0 else None,
        },
        "memory": memory,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
