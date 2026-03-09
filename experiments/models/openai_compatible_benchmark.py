from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from leanstack.prompt_bucket import build_exact_prompt_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible completion endpoint.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--variant", default="openai")
    parser.add_argument("--benchmark-profile", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer-model-path", default="")
    parser.add_argument("--exact-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output")
    return parser.parse_args()


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), dict(resp.headers), resp.read()


def run_streaming_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{_normalize_base_url(base_url)}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token_seconds: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    chunks: list[str] = []

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if usage is None and event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("text") or ""
                if delta:
                    if first_token_seconds is None:
                        first_token_seconds = time.perf_counter() - start
                    chunks.append(delta)
                finish_reason = choice.get("finish_reason") or finish_reason

    elapsed = time.perf_counter() - start
    return {
        "generated_text": "".join(chunks),
        "ttft_seconds": first_token_seconds,
        "stream_seconds": elapsed,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def run_non_streaming_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }
    start = time.perf_counter()
    _, _, body = _post_json(
        f"{_normalize_base_url(base_url)}/v1/completions",
        payload=payload,
        api_key=api_key,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response = json.loads(body.decode("utf-8"))
    choice = (response.get("choices") or [{}])[0]
    return {
        "generated_text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "end_to_end_seconds": elapsed,
    }


def resolve_usage(
    stream_result: dict[str, Any],
    non_stream_result: dict[str, Any],
) -> dict[str, Any] | None:
    usage = stream_result.get("usage")
    if usage is not None:
        return usage
    return non_stream_result.get("usage")


def resolve_prompt(args: argparse.Namespace) -> tuple[str, int | None]:
    if args.exact_prompt_tokens <= 0:
        return args.prompt, None
    if not args.tokenizer_model_path:
        raise ValueError("--tokenizer-model-path is required when --exact-prompt-tokens is set")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_path, trust_remote_code=True)
    exact_prompt, exact_tokens = build_exact_prompt_text(tokenizer, args.prompt, args.exact_prompt_tokens)
    return exact_prompt, exact_tokens


def main() -> int:
    args = parse_args()
    prompt, target_prompt_tokens = resolve_prompt(args)
    try:
        stream_result = run_streaming_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
        non_stream_result = run_non_streaming_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            api_key=args.api_key,
            timeout=args.request_timeout,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc

    usage = resolve_usage(stream_result, non_stream_result) or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    end_to_end_seconds = non_stream_result["end_to_end_seconds"]

    result = {
        "system": args.system,
        "variant": args.variant,
        "benchmark_profile": args.benchmark_profile or None,
        "base_url": _normalize_base_url(args.base_url),
        "model": args.model,
        "prompt": prompt,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": completion_tokens,
        "ttft_seconds": stream_result["ttft_seconds"],
        "stream_seconds": stream_result["stream_seconds"],
        "end_to_end_seconds": end_to_end_seconds,
        "generated_tokens_per_second": (
            (completion_tokens / stream_result["stream_seconds"])
            if completion_tokens is not None and stream_result["stream_seconds"] > 0
            else None
        ),
        "end_to_end_tokens_per_second": (
            (completion_tokens / end_to_end_seconds) if completion_tokens is not None and end_to_end_seconds > 0 else None
        ),
        "finish_reason": stream_result["finish_reason"] or non_stream_result["finish_reason"],
        "generated_text": non_stream_result["generated_text"] or stream_result["generated_text"],
        "usage_source": "stream" if stream_result.get("usage") is not None else "non_stream_fallback",
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
