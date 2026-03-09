from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from leanstack.prompt_bucket import build_exact_prompt_text
from leanstack.leanserve import (
    materialize_qwen_full_semantic_runtime_from_leanpack,
    materialize_qwen_semantic_stack_from_leanpack,
)
from leanstack.runtime.kv_cache import StaticKVBlockManager
from leanstack.runtime.qwen_explicit import (
    build_qwen_position_cache,
    materialize_qwen_full_runtime,
    materialize_qwen_full_semantic_runtime,
    materialize_qwen_stack_runtime,
    materialize_qwen_semantic_stack_runtime,
    normalize_stop_token_ids,
    project_hidden_to_logits,
    resolve_semantic_logits_backend,
    run_semantic_stack_decode_from_hidden,
    run_semantic_stack_decode,
    run_semantic_stack_prefill_from_hidden,
    run_semantic_stack_prefill,
    run_stack_decode,
    run_stack_prefill,
    select_semantic_greedy_token,
    select_greedy_token,
    try_compile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an explicit Qwen3 runtime loop on GPU.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pack-dir", default="", help="Optional leanpack artifact directory for semantic runtime.")
    parser.add_argument("--runtime-mode", choices=("borrowed", "semantic"), default="borrowed")
    parser.add_argument("--benchmark-profile", default="")
    parser.add_argument("--num-layers", type=int, default=0, help="0 means the full model.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prompt", default="Explain why fixing the model-chip contract simplifies inference software.")
    parser.add_argument("--prompt-format", choices=("auto", "chat", "raw"), default="chat")
    parser.add_argument("--exact-prefill-bucket", action="store_true")
    parser.add_argument("--max-prefill-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--speculative", action="store_true", help="Enable exact self-speculative decode.")
    parser.add_argument("--draft-layer-count", type=int, default=12)
    parser.add_argument("--proposal-len", type=int, default=4)
    parser.add_argument("--resident-requests", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--capture-decode-step-timings", action="store_true")
    parser.add_argument("--stop-token-id", action="append", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--skip-final-cache-advance", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the decode hot path.")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
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


def build_prompt_input_ids(tokenizer, prompt_text: str, args: argparse.Namespace) -> tuple[torch.LongTensor, str, int | None]:
    exact_prompt_text = prompt_text
    target_prompt_tokens: int | None = None
    if args.exact_prefill_bucket:
        target_prompt_tokens = args.max_prefill_tokens
        exact_prompt_text, _ = build_exact_prompt_text(tokenizer, prompt_text, target_prompt_tokens)
    input_ids = tokenizer(exact_prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][
        :, : args.max_prefill_tokens
    ]
    return input_ids, exact_prompt_text, target_prompt_tokens


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


def materialize_runtime(args: argparse.Namespace):
    if args.runtime_mode == "semantic":
        if args.pack_dir:
            if args.num_layers == 0:
                return materialize_qwen_full_semantic_runtime_from_leanpack(
                    args.pack_dir,
                    device=args.device,
                    dtype=args.dtype,
                    include_output_head=True,
                )
            return materialize_qwen_semantic_stack_from_leanpack(
                args.pack_dir,
                layer_indices=tuple(range(args.num_layers)),
                device=args.device,
                dtype=args.dtype,
                include_output_head=True,
            )
        if args.num_layers == 0:
            return materialize_qwen_full_semantic_runtime(
                model_path=args.model_path,
                device=args.device,
                dtype=args.dtype,
                include_output_head=True,
            )
        return materialize_qwen_semantic_stack_runtime(
            model_path=args.model_path,
            layer_indices=tuple(range(args.num_layers)),
            device=args.device,
            dtype=args.dtype,
            include_output_head=True,
        )

    if args.num_layers == 0:
        return materialize_qwen_full_runtime(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            include_output_head=True,
        )
    return materialize_qwen_stack_runtime(
        model_path=args.model_path,
        layer_indices=tuple(range(args.num_layers)),
        device=args.device,
        dtype=args.dtype,
        include_output_head=True,
    )


def materialize_speculative_runtimes(args: argparse.Namespace):
    if args.runtime_mode != "semantic":
        raise ValueError("--speculative currently requires --runtime-mode semantic")
    if not args.pack_dir:
        raise ValueError("--speculative currently requires --pack-dir")
    if args.draft_layer_count <= 0:
        raise ValueError("--draft-layer-count must be positive")

    draft_runtime = materialize_qwen_semantic_stack_from_leanpack(
        args.pack_dir,
        layer_indices=tuple(range(args.draft_layer_count)),
        device=args.device,
        dtype=args.dtype,
        include_output_head=True,
    )
    total_layers = int(draft_runtime.config.num_hidden_layers)
    if args.draft_layer_count >= total_layers:
        raise ValueError(
            f"--draft-layer-count must be smaller than total layers ({total_layers}), got {args.draft_layer_count}"
        )
    verifier_runtime = materialize_qwen_semantic_stack_from_leanpack(
        args.pack_dir,
        layer_indices=tuple(range(args.draft_layer_count, total_layers)),
        device=args.device,
        dtype=args.dtype,
        include_output_head=True,
    )
    return draft_runtime, verifier_runtime


def select_borrowed_greedy_token(runtime, hidden_states: torch.Tensor) -> torch.LongTensor:
    return select_greedy_token(project_hidden_to_logits(runtime, hidden_states))


def run_request(
    *,
    runtime,
    tokenizer,
    input_ids: torch.LongTensor,
    device: torch.device,
    runtime_mode: str,
    page_size: int,
    max_new_tokens: int,
    skip_final_cache_advance: bool,
    stop_token_ids: tuple[int, ...],
    capture_decode_step_timings: bool,
    decode_fn,
    select_fn,
) -> dict:
    generated_gpu = torch.empty((max_new_tokens,), device=device, dtype=torch.long)
    decode_step_seconds: list[float] = []
    stop_reason = "max_new_tokens"
    timings: dict[str, float | list[float]] = {}
    position_cache = None
    stop_token_tensor = None
    if runtime_mode == "semantic":
        position_cache = build_qwen_position_cache(
            runtime.rope_inv_freq,
            runtime.attention_scaling,
            max_seq_len=int(input_ids.shape[-1]) + max_new_tokens,
            dtype=runtime.dtype,
        )
    if stop_token_ids:
        stop_token_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)
    emitted_tokens = 0

    if runtime_mode == "semantic":
        def prefill_once():
            prefill_hidden, cache_state = run_semantic_stack_prefill(
                runtime,
                input_ids,
                page_size=page_size,
                max_seq_len=int(input_ids.shape[-1]) + max_new_tokens,
                position_cache=position_cache,
            )
            return select_fn(runtime, prefill_hidden), cache_state

        def decode_once(token: torch.LongTensor, cache_state):
            decode_hidden, cache_state = decode_fn(
                runtime,
                token,
                cache_state,
                position_cache=position_cache,
            )
            return select_fn(runtime, decode_hidden), cache_state
    else:
        def prefill_once():
            prefill_hidden, cache_state = run_stack_prefill(runtime, input_ids)
            return select_fn(runtime, prefill_hidden), cache_state

        def decode_once(token: torch.LongTensor, cache_state):
            decode_hidden, cache_state = decode_fn(runtime, token, cache_state)
            return select_fn(runtime, decode_hidden), cache_state

    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        next_token, cache = prefill_once()
        sync_device(device)
        timings["prefill_seconds"] = time.perf_counter() - start

        if capture_decode_step_timings and device.type == "cuda":
            loop_start_event = torch.cuda.Event(enable_timing=True)
            loop_end_event = torch.cuda.Event(enable_timing=True)
            loop_start_event.record()
            step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        else:
            sync_device(device)
            loop_start = time.perf_counter()
        if stop_token_tensor is None:
            if max_new_tokens > 0:
                for step in range(max_new_tokens - 1):
                    generated_gpu[step] = next_token.reshape(())
                    if capture_decode_step_timings and device.type == "cuda":
                        step_start_event = torch.cuda.Event(enable_timing=True)
                        step_end_event = torch.cuda.Event(enable_timing=True)
                        step_start_event.record()
                    next_token, cache = decode_once(next_token, cache)
                    if capture_decode_step_timings and device.type == "cuda":
                        step_end_event.record()
                        step_events.append((step_start_event, step_end_event))
                generated_gpu[max_new_tokens - 1] = next_token.reshape(())
                emitted_tokens = max_new_tokens
        else:
            for step in range(max_new_tokens):
                generated_gpu[step] = next_token.reshape(())
                emitted_tokens = step + 1
                should_stop = False

                stop_hit = torch.eq(next_token.reshape(1), stop_token_tensor).any()
                if bool(stop_hit.item()):
                    stop_reason = "stop_token"
                    should_stop = True

                if step == max_new_tokens - 1:
                    should_stop = True

                if should_stop:
                    break

                if capture_decode_step_timings and device.type == "cuda":
                    step_start_event = torch.cuda.Event(enable_timing=True)
                    step_end_event = torch.cuda.Event(enable_timing=True)
                    step_start_event.record()
                next_token, cache = decode_once(next_token, cache)
                if capture_decode_step_timings and device.type == "cuda":
                    step_end_event.record()
                    step_events.append((step_start_event, step_end_event))

        if capture_decode_step_timings and device.type == "cuda":
            loop_end_event.record()
            sync_device(device)
            timings["decode_loop_seconds"] = float(loop_start_event.elapsed_time(loop_end_event)) / 1000.0
            decode_step_seconds = [
                float(start_event.elapsed_time(end_event)) / 1000.0 for start_event, end_event in step_events
            ]
        else:
            sync_device(device)
            timings["decode_loop_seconds"] = time.perf_counter() - loop_start

        timings["final_cache_advance_seconds"] = 0.0
        final_cache_advanced = False
        if emitted_tokens > 0 and not skip_final_cache_advance:
            final_token = generated_gpu[emitted_tokens - 1].view(1, 1)
            sync_device(device)
            start = time.perf_counter()
            if runtime_mode == "semantic":
                _, cache = run_semantic_stack_decode(runtime, final_token, cache, position_cache=position_cache)
            else:
                _, cache = run_stack_decode(runtime, final_token, cache)
            sync_device(device)
            timings["final_cache_advance_seconds"] = time.perf_counter() - start
            final_cache_advanced = True

    generated_tensor = generated_gpu[:emitted_tokens].unsqueeze(0).to("cpu")
    runtime_loop_seconds = float(timings["prefill_seconds"]) + float(timings["decode_loop_seconds"])
    full_loop_seconds = runtime_loop_seconds + float(timings["final_cache_advance_seconds"])
    generated_ids = generated_tensor[0].tolist()
    return {
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_tensor[0], skip_special_tokens=True),
        "cache_seq_length": int(cache.get_seq_length()),
        "cache_summary": cache.summary() if runtime_mode == "semantic" else None,
        "final_cache_advanced": final_cache_advanced,
        "decode_steps_executed": len(decode_step_seconds),
        "stop_reason": stop_reason,
        "emitted_tokens": emitted_tokens,
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
    }


def run_speculative_request(
    *,
    draft_runtime,
    verifier_runtime,
    tokenizer,
    input_ids: torch.LongTensor,
    device: torch.device,
    page_size: int,
    max_new_tokens: int,
    proposal_len: int,
    stop_token_ids: tuple[int, ...],
) -> dict:
    if proposal_len <= 0:
        raise ValueError("proposal_len must be positive")

    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    timings: dict[str, float | list[float]] = {}
    speculative_summary = {
        "draft_layer_count": len(draft_runtime.layer_indices),
        "verifier_layer_count": len(verifier_runtime.layer_indices),
        "proposal_len": proposal_len,
        "cycles": 0,
        "proposed_tokens": 0,
        "accepted_proposal_tokens": 0,
        "rejected_proposal_tokens": 0,
        "bonus_tokens": 0,
        "accepted_tokens_per_cycle": [],
    }
    stop_token_set = set(stop_token_ids)
    max_seq_len = int(input_ids.shape[-1]) + max_new_tokens + proposal_len + 1
    position_cache = build_qwen_position_cache(
        draft_runtime.rope_inv_freq,
        draft_runtime.attention_scaling,
        max_seq_len=max_seq_len,
        dtype=draft_runtime.dtype,
    )

    with torch.inference_mode():
        sync_device(device)
        start = time.perf_counter()
        draft_hidden, draft_cache = run_semantic_stack_prefill(
            draft_runtime,
            input_ids,
            page_size=page_size,
            max_seq_len=max_seq_len,
            position_cache=position_cache,
        )
        verifier_hidden, verifier_cache = run_semantic_stack_prefill_from_hidden(
            verifier_runtime,
            draft_hidden,
            page_size=page_size,
            max_seq_len=max_seq_len,
            position_cache=position_cache,
        )
        if not isinstance(draft_cache, StaticKVBlockManager) or not isinstance(verifier_cache, StaticKVBlockManager):
            raise TypeError("speculative path currently requires StaticKVBlockManager caches")
        current_draft_hidden = draft_hidden
        current_exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden)
        sync_device(device)
        timings["prefill_seconds"] = time.perf_counter() - start

        sync_device(device)
        loop_start = time.perf_counter()
        while len(generated_ids) < max_new_tokens:
            speculative_summary["cycles"] += 1
            remaining_tokens = max_new_tokens - len(generated_ids)
            cycle_proposal_len = min(proposal_len, remaining_tokens)
            base_snapshot = draft_cache.snapshot_state()
            draft_snapshots = [base_snapshot]
            proposed_ids: list[int] = []
            proposed_hiddens: list[torch.Tensor] = []
            draft_hidden_cursor = current_draft_hidden

            for _ in range(cycle_proposal_len):
                draft_token = select_semantic_greedy_token(draft_runtime, draft_hidden_cursor)
                proposed_ids.append(int(draft_token.item()))
                draft_hidden_cursor, draft_cache = run_semantic_stack_decode(
                    draft_runtime,
                    draft_token,
                    draft_cache,
                    position_cache=position_cache,
                )
                proposed_hiddens.append(draft_hidden_cursor)
                draft_snapshots.append(draft_cache.snapshot_state())

            speculative_summary["proposed_tokens"] += len(proposed_ids)
            exact_token = current_exact_token
            accepted_this_cycle = 0
            cycle_committed: list[int] = []
            mismatch = False

            for proposal_idx, proposal_id in enumerate(proposed_ids):
                if proposal_id != int(exact_token.item()):
                    mismatch = True
                    speculative_summary["rejected_proposal_tokens"] += len(proposed_ids) - proposal_idx
                    cycle_committed.append(int(exact_token.item()))
                    draft_cache.restore_state(draft_snapshots[proposal_idx])
                    exact_token_for_draft = exact_token.clone()
                    current_draft_hidden, draft_cache = run_semantic_stack_decode(
                        draft_runtime,
                        exact_token_for_draft,
                        draft_cache,
                        position_cache=position_cache,
                    )
                    verifier_hidden_exact, verifier_cache = run_semantic_stack_decode_from_hidden(
                        verifier_runtime,
                        current_draft_hidden,
                        verifier_cache,
                        position_cache=position_cache,
                    )
                    current_exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden_exact)
                    break

                cycle_committed.append(proposal_id)
                accepted_this_cycle += 1
                speculative_summary["accepted_proposal_tokens"] += 1
                current_draft_hidden = proposed_hiddens[proposal_idx]
                verifier_hidden_accept, verifier_cache = run_semantic_stack_decode_from_hidden(
                    verifier_runtime,
                    current_draft_hidden,
                    verifier_cache,
                    position_cache=position_cache,
                )
                exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden_accept)

            if not mismatch:
                current_exact_token = exact_token
                if len(generated_ids) + len(cycle_committed) < max_new_tokens:
                    bonus_token = current_exact_token.clone()
                    cycle_committed.append(int(bonus_token.item()))
                    speculative_summary["bonus_tokens"] += 1
                    current_draft_hidden, draft_cache = run_semantic_stack_decode(
                        draft_runtime,
                        bonus_token,
                        draft_cache,
                        position_cache=position_cache,
                    )
                    verifier_hidden_bonus, verifier_cache = run_semantic_stack_decode_from_hidden(
                        verifier_runtime,
                        current_draft_hidden,
                        verifier_cache,
                        position_cache=position_cache,
                    )
                    current_exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden_bonus)

            emitted_this_cycle = 0
            for token_id in cycle_committed:
                if len(generated_ids) >= max_new_tokens:
                    break
                generated_ids.append(int(token_id))
                emitted_this_cycle += 1
                if stop_token_set and token_id in stop_token_set:
                    stop_reason = "stop_token"
                    break
            speculative_summary["accepted_tokens_per_cycle"].append(emitted_this_cycle)
            if stop_reason == "stop_token":
                break

        sync_device(device)
        timings["decode_loop_seconds"] = time.perf_counter() - loop_start
        timings["final_cache_advance_seconds"] = 0.0

    runtime_loop_seconds = float(timings["prefill_seconds"]) + float(timings["decode_loop_seconds"])
    full_loop_seconds = runtime_loop_seconds
    cycle_count = int(speculative_summary["cycles"])
    accepted_proposals = int(speculative_summary["accepted_proposal_tokens"])
    proposed_tokens = int(speculative_summary["proposed_tokens"])
    speculative_summary["acceptance_ratio"] = (
        float(accepted_proposals / proposed_tokens) if proposed_tokens > 0 else None
    )
    speculative_summary["committed_tokens_per_cycle"] = (
        float(len(generated_ids) / cycle_count) if cycle_count > 0 else None
    )

    return {
        "generated_token_ids": list(generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "cache_seq_length": int(verifier_cache.get_seq_length()),
        "cache_summary": {
            "draft": draft_cache.summary(),
            "verifier": verifier_cache.summary(),
        },
        "final_cache_advanced": False,
        "decode_steps_executed": 0,
        "stop_reason": stop_reason,
        "emitted_tokens": len(generated_ids),
        "timings": {
            **timings,
            "runtime_loop_seconds": runtime_loop_seconds,
            "full_loop_seconds": full_loop_seconds,
            "decode_step_seconds": [],
        },
        "throughput": {
            "runtime_tokens_per_second": (len(generated_ids) / runtime_loop_seconds) if runtime_loop_seconds > 0 else None,
            "full_loop_tokens_per_second": (len(generated_ids) / full_loop_seconds) if full_loop_seconds > 0 else None,
        },
        "speculative_summary": speculative_summary,
    }


def average_request_metrics(requests: list[dict]) -> dict:
    if not requests:
        raise ValueError("expected at least one measured request")

    def average(values: list[float | None]) -> float | None:
        concrete = [float(value) for value in values if value is not None]
        if not concrete:
            return None
        return sum(concrete) / len(concrete)

    avg_timings = {
        "prefill_seconds": average([request["timings"]["prefill_seconds"] for request in requests]),
        "decode_loop_seconds": average([request["timings"]["decode_loop_seconds"] for request in requests]),
        "final_cache_advance_seconds": average([request["timings"]["final_cache_advance_seconds"] for request in requests]),
        "runtime_loop_seconds": average([request["timings"]["runtime_loop_seconds"] for request in requests]),
        "full_loop_seconds": average([request["timings"]["full_loop_seconds"] for request in requests]),
        "decode_step_seconds": requests[-1]["timings"]["decode_step_seconds"],
    }
    avg_throughput = {
        "runtime_tokens_per_second": average(
            [request["throughput"]["runtime_tokens_per_second"] for request in requests]
        ),
        "full_loop_tokens_per_second": average([request["throughput"]["full_loop_tokens_per_second"] for request in requests]),
    }
    return {
        "timings": avg_timings,
        "throughput": avg_throughput,
        "emitted_tokens": average([request["emitted_tokens"] for request in requests]),
    }


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.num_layers < 0:
        raise ValueError("--num-layers must be non-negative")
    if args.resident_requests <= 0:
        raise ValueError("--resident-requests must be positive")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    if args.proposal_len <= 0:
        raise ValueError("--proposal-len must be positive")
    if args.speculative and args.capture_decode_step_timings:
        raise ValueError("--capture-decode-step-timings is not supported with --speculative")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    thinking_mode = resolve_thinking_mode(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    formatted_prompt, resolved_prompt_format = build_prompt(tokenizer, args, thinking_mode)
    input_ids, exact_prompt_text, target_prompt_tokens = build_prompt_input_ids(tokenizer, formatted_prompt, args)
    stop_token_ids = resolve_stop_token_ids(tokenizer, args)

    timings: dict[str, float] = {}
    memory: dict[str, dict[str, int] | None] = {"before_materialize": snapshot_cuda_memory(device)}

    sync_device(device)
    start = time.perf_counter()
    if args.speculative:
        draft_runtime, verifier_runtime = materialize_speculative_runtimes(args)
        runtime = verifier_runtime
    else:
        runtime = materialize_runtime(args)
    sync_device(device)
    timings["materialize_seconds"] = time.perf_counter() - start
    memory["after_materialize"] = snapshot_cuda_memory(device)

    if args.runtime_mode == "semantic":
        decode_fn = run_semantic_stack_decode
        select_fn = select_semantic_greedy_token
    else:
        decode_fn = run_stack_decode
        select_fn = select_borrowed_greedy_token

    compiled_decode = False
    compiled_select = False
    if args.compile and not args.speculative:
        compiled_decode_fn = try_compile(decode_fn, mode=args.compile_mode)
        compiled_select_fn = try_compile(select_fn, mode=args.compile_mode)
        compiled_decode = compiled_decode_fn is not decode_fn
        compiled_select = compiled_select_fn is not select_fn
        decode_fn = compiled_decode_fn
        select_fn = compiled_select_fn

    warmup_results: list[dict] = []
    measured_results: list[dict] = []
    total_requests = args.warmup_requests + args.resident_requests
    for request_idx in range(total_requests):
        if args.speculative:
            request_result = run_speculative_request(
                draft_runtime=draft_runtime,
                verifier_runtime=verifier_runtime,
                tokenizer=tokenizer,
                input_ids=input_ids.to(device),
                device=device,
                page_size=args.page_size,
                max_new_tokens=args.max_new_tokens,
                proposal_len=args.proposal_len,
                stop_token_ids=stop_token_ids,
            )
        else:
            request_result = run_request(
                runtime=runtime,
                tokenizer=tokenizer,
                input_ids=input_ids.to(device),
                device=device,
                runtime_mode=args.runtime_mode,
                page_size=args.page_size,
                max_new_tokens=args.max_new_tokens,
                skip_final_cache_advance=args.skip_final_cache_advance,
                stop_token_ids=stop_token_ids,
                capture_decode_step_timings=args.capture_decode_step_timings,
                decode_fn=decode_fn,
                select_fn=select_fn,
            )
        if request_idx < args.warmup_requests:
            warmup_results.append(request_result)
        else:
            measured_results.append(request_result)

    aggregate = average_request_metrics(measured_results)
    last_result = measured_results[-1]
    memory["after_runtime_loop"] = snapshot_cuda_memory(device)

    result = {
        "model_path": args.model_path,
        "num_layers": int(runtime.config.num_hidden_layers) if args.speculative else len(runtime.layer_indices),
        "layer_range": (
            [0, int(runtime.config.num_hidden_layers) - 1]
            if args.speculative
            else [runtime.layer_indices[0], runtime.layer_indices[-1]] if runtime.layer_indices else []
        ),
        "runtime_mode": args.runtime_mode,
        "semantic_logits_backend": resolve_semantic_logits_backend() if args.runtime_mode == "semantic" else None,
        "benchmark_profile": args.benchmark_profile or None,
        "device": args.device,
        "dtype": args.dtype,
        "compiled_decode": compiled_decode,
        "compiled_select": compiled_select,
        "compile_mode": args.compile_mode if args.compile else None,
        "attn_implementation": runtime.config._attn_implementation,
        "speculative": bool(args.speculative),
        "draft_layer_count": args.draft_layer_count if args.speculative else None,
        "proposal_len": args.proposal_len if args.speculative else None,
        "prompt_format": resolved_prompt_format,
        "exact_prefill_bucket": bool(args.exact_prefill_bucket),
        "target_prompt_tokens": target_prompt_tokens,
        "thinking_mode": (
            "enabled" if thinking_mode is True else "disabled" if thinking_mode is False else "default"
        ),
        "prompt_text": exact_prompt_text,
        "prompt_tokens": int(input_ids.shape[-1]),
        "max_new_tokens": args.max_new_tokens,
        "resident_requests": args.resident_requests,
        "warmup_requests": args.warmup_requests,
        "emitted_tokens": int(last_result["emitted_tokens"]),
        "decode_steps_executed": int(last_result["decode_steps_executed"]),
        "stop_reason": last_result["stop_reason"],
        "stop_token_ids": list(stop_token_ids),
        "generated_token_ids": last_result["generated_token_ids"],
        "generated_text": last_result["generated_text"],
        "cache_seq_length": int(last_result["cache_seq_length"]),
        "cache_summary": last_result["cache_summary"],
        "final_cache_advanced": bool(last_result["final_cache_advanced"]),
        "timings": {**timings, **aggregate["timings"]},
        "throughput": aggregate["throughput"],
        "resident_summary": {
            "measured_requests": args.resident_requests,
            "warmup_requests": args.warmup_requests,
            "measured_runtime_tokens_per_second": [
                request["throughput"]["runtime_tokens_per_second"] for request in measured_results
            ],
            "measured_prefill_seconds": [request["timings"]["prefill_seconds"] for request in measured_results],
            "warmup_runtime_tokens_per_second": [
                request["throughput"]["runtime_tokens_per_second"] for request in warmup_results
            ],
        },
        "speculative_summary": last_result.get("speculative_summary"),
        "memory": memory,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
