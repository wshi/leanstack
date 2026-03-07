from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from leanstack.runtime.qwen_explicit import (
    materialize_qwen_stack_runtime,
    project_hidden_to_logits,
    run_stack_decode,
    run_stack_forward,
    run_stack_prefill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an explicit multi-layer Qwen3 stack on GPU.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prompt", default="Explain why fixing the model-chip contract simplifies inference software.")
    parser.add_argument("--prompt-format", choices=("auto", "chat", "raw"), default="chat")
    parser.add_argument("--max-prefill-tokens", type=int, default=16)
    parser.add_argument("--with-output-head", action="store_true")
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


def main() -> int:
    args = parse_args()
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be positive")

    thinking_mode = resolve_thinking_mode(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    formatted_prompt, resolved_prompt_format = build_prompt(tokenizer, args, thinking_mode)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][:, : args.max_prefill_tokens]
    decode_ids = input_ids[:, -1:].clone()
    layer_indices = tuple(range(args.num_layers))

    timings: dict[str, float] = {}

    start = time.perf_counter()
    runtime = materialize_qwen_stack_runtime(
        model_path=args.model_path,
        layer_indices=layer_indices,
        device=args.device,
        dtype=args.dtype,
        include_output_head=args.with_output_head,
    )
    timings["materialize_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    forward_hidden = run_stack_forward(runtime, input_ids)
    timings["forward_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    prefill_hidden, cache = run_stack_prefill(runtime, input_ids)
    timings["prefill_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    decode_hidden, cache = run_stack_decode(runtime, decode_ids, cache)
    timings["decode_seconds"] = time.perf_counter() - start

    result = {
        "model_path": args.model_path,
        "num_layers": args.num_layers,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": runtime.config._attn_implementation,
        "prompt_format": resolved_prompt_format,
        "thinking_mode": (
            "enabled" if thinking_mode is True else "disabled" if thinking_mode is False else "default"
        ),
        "prefill_tokens": int(input_ids.shape[-1]),
        "decode_tokens": int(decode_ids.shape[-1]),
        "forward_hidden_shape": list(forward_hidden.shape),
        "prefill_hidden_shape": list(prefill_hidden.shape),
        "decode_hidden_shape": list(decode_hidden.shape),
        "cache_seq_length": int(cache.get_seq_length()),
        "timings": timings,
    }

    if args.with_output_head:
        prefill_logits = project_hidden_to_logits(runtime, prefill_hidden)
        decode_logits = project_hidden_to_logits(runtime, decode_hidden)
        result["prefill_logits_shape"] = list(prefill_logits.shape)
        result["decode_logits_shape"] = list(decode_logits.shape)
        result["decode_top_token_id"] = int(decode_logits[:, -1, :].argmax(dim=-1).item())

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
