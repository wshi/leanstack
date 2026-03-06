from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline Hugging Face GLM smoke test.")
    parser.add_argument("--model-id", default="zai-org/glm-4-9b-hf")
    parser.add_argument("--prompt", default="Summarize why a TileIR-first serving stack is useful.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    result = {
        "model_id": args.model_id,
        "device": str(model_device),
        "dtype": args.dtype,
        "prompt": args.prompt,
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

