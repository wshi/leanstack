from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline Hugging Face causal LM smoke test.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--prompt",
        default="Explain why agent-built, model-chip-specific software may beat a compatibility-heavy serving stack.",
    )
    parser.add_argument("--prompt-format", choices=("auto", "chat", "raw"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def resolve_thinking_mode(args: argparse.Namespace) -> bool | None:
    if args.enable_thinking:
        return True
    if args.disable_thinking:
        return False
    return None


def build_prompt(tokenizer, args: argparse.Namespace, thinking_mode: bool | None) -> tuple[str, str]:
    use_chat = False
    if args.prompt_format == "chat":
        use_chat = True
    elif args.prompt_format == "auto":
        use_chat = getattr(tokenizer, "chat_template", None) is not None

    if not use_chat:
        return args.prompt, "raw"

    messages = [{"role": "user", "content": args.prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if thinking_mode is not None:
        template_kwargs["enable_thinking"] = thinking_mode

    try:
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
    return formatted, "chat"


def main() -> int:
    args = parse_args()
    thinking_mode = resolve_thinking_mode(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model = model.to(args.device)

    formatted_prompt, resolved_prompt_format = build_prompt(tokenizer, args, thinking_mode)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    model_device = torch.device(args.device)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = args.temperature

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start

    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tokens = int(generated.shape[-1])

    result = {
        "model_id": args.model_id,
        "device": str(model_device),
        "dtype": args.dtype,
        "prompt": args.prompt,
        "prompt_format": resolved_prompt_format,
        "thinking_mode": (
            "enabled" if thinking_mode is True else "disabled" if thinking_mode is False else "default"
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed if elapsed > 0 else None,
        "generated_text": text,
        "elapsed_seconds": elapsed,
    }
    payload = json.dumps(result, indent=2)
    print(payload)

    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
