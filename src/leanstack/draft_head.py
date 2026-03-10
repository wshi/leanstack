from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from .leanserve import materialize_qwen_semantic_stack_from_leanpack
from .pack import DraftHeadEntry, PackedFileEntry, PackedTensorEntry, load_packed_artifact_manifest
from .prompt_bucket import build_exact_prompt_text
from .runtime.qwen_explicit import (
    build_qwen_position_cache,
    run_semantic_stack_decode,
    run_semantic_stack_decode_from_hidden,
    run_semantic_stack_forward,
    run_semantic_stack_forward_from_hidden,
    run_semantic_stack_prefill,
    run_semantic_stack_prefill_from_hidden,
    select_semantic_greedy_token,
)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float64, torch.int64):
        return 8
    if dtype in (torch.int8, torch.uint8, torch.bool):
        return 1
    raise ValueError(f"unsupported dtype: {dtype}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class DraftHeadFitResult:
    key: str
    draft_layer_count: int
    calibration_mode: str
    fit_samples: int
    chunk_tokens: int
    chunks_used: int
    decode_steps: int
    ridge_lambda: float
    projection_shape: tuple[int, int]
    corpus_tokens: int
    calibration_hidden_mse_before: float
    calibration_hidden_mse_after: float

    def as_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "draft_layer_count": self.draft_layer_count,
            "calibration_mode": self.calibration_mode,
            "fit_samples": self.fit_samples,
            "chunk_tokens": self.chunk_tokens,
            "chunks_used": self.chunks_used,
            "decode_steps": self.decode_steps,
            "ridge_lambda": self.ridge_lambda,
            "projection_shape": list(self.projection_shape),
            "corpus_tokens": self.corpus_tokens,
            "calibration_hidden_mse_before": self.calibration_hidden_mse_before,
            "calibration_hidden_mse_after": self.calibration_hidden_mse_after,
        }


def _repo_default_corpus(repo_root: Path) -> str:
    pieces: list[str] = []
    for path in [repo_root / "README.md", *sorted((repo_root / "docs").glob("*.md"))]:
        if path.exists():
            pieces.append(path.read_text(encoding="utf-8", errors="ignore"))
    corpus = "\n\n".join(pieces).strip()
    if not corpus:
        raise FileNotFoundError("could not find README.md/docs corpus for draft-head calibration")
    return corpus


def _build_token_chunks(
    tokenizer,
    corpus_text: str,
    *,
    chunk_tokens: int,
    max_chunks: int,
) -> tuple[list[torch.LongTensor], int]:
    token_ids = tokenizer(corpus_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if token_ids.numel() < chunk_tokens:
        repeats = (chunk_tokens + token_ids.numel() - 1) // token_ids.numel()
        token_ids = token_ids.repeat(repeats)
    chunks: list[torch.LongTensor] = []
    offset = 0
    while offset + chunk_tokens <= token_ids.numel() and len(chunks) < max_chunks:
        chunks.append(token_ids[offset : offset + chunk_tokens].unsqueeze(0).contiguous())
        offset += chunk_tokens
    return chunks, int(token_ids.numel())


def _build_prompt_chunks(
    tokenizer,
    corpus_text: str,
    *,
    prompt_tokens: int,
    max_prompts: int,
) -> tuple[list[torch.LongTensor], int]:
    prompts: list[torch.LongTensor] = []
    corpus_tokens = 0
    segments = [segment.strip() for segment in corpus_text.split("\n\n") if segment.strip()]
    if not segments:
        raise ValueError("could not derive prompt segments from corpus")
    for seed in segments:
        exact_prompt_text, _ = build_exact_prompt_text(tokenizer, seed, prompt_tokens)
        input_ids = tokenizer(exact_prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][
            :, :prompt_tokens
        ].contiguous()
        if input_ids.shape[-1] != prompt_tokens:
            continue
        prompts.append(input_ids)
        corpus_tokens += int(input_ids.numel())
        if len(prompts) >= max_prompts:
            break
    if not prompts:
        raise ValueError("failed to build any exact-bucket prompt chunks")
    return prompts, corpus_tokens


def _fit_linear_projection(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    x = draft_hidden.to(torch.float32)
    y = target_hidden.to(torch.float32)
    xtx = x.transpose(0, 1) @ x
    if ridge_lambda > 0:
        xtx = xtx + (ridge_lambda * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device))
    xty = x.transpose(0, 1) @ y
    solution = torch.linalg.solve(xtx, xty)
    return solution.transpose(0, 1).contiguous()


def _load_safetensor_file(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        return {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).contiguous() for name in handle.keys()}


def _update_manifest_with_draft_head(
    *,
    pack_dir: Path,
    key: str,
    draft_layer_count: int,
    projection_weight: torch.Tensor,
    fit_result: DraftHeadFitResult,
) -> None:
    manifest = load_packed_artifact_manifest(pack_dir)
    draft_head_file = "draft-heads.safetensors"
    tensor_name = f"draft_heads.layer{draft_layer_count}.{key}.projection.weight"
    file_path = pack_dir / draft_head_file

    tensors = _load_safetensor_file(file_path)
    tensors[tensor_name] = projection_weight.detach().cpu().contiguous()
    save_file(tensors, str(file_path))
    size_bytes = int(file_path.stat().st_size)

    updated_head = DraftHeadEntry(
        key=key,
        draft_layer_count=draft_layer_count,
        file=draft_head_file,
        tensor_name=tensor_name,
        input_size=int(projection_weight.shape[1]),
        output_size=int(projection_weight.shape[0]),
        fit_samples=fit_result.fit_samples,
        ridge_lambda=fit_result.ridge_lambda,
    )
    draft_heads = [head for head in manifest.draft_heads if head.key != key]
    draft_heads.append(updated_head)

    files = [file_entry for file_entry in manifest.files if file_entry.file != draft_head_file]
    files.append(
        PackedFileEntry(
            file=draft_head_file,
            tensor_count=len(tensors),
            size_bytes=size_bytes,
        )
    )

    tensor_entries = [entry for entry in manifest.tensors if entry.file != draft_head_file]
    logical_offset = 0
    for name, tensor in tensors.items():
        size = int(tensor.numel()) * _dtype_nbytes(tensor.dtype)
        tensor_entries.append(
            PackedTensorEntry(
                name=name,
                role="draft_head_projection",
                file=draft_head_file,
                dtype=_dtype_name(tensor.dtype),
                shape=list(tensor.shape),
                numel=int(tensor.numel()),
                logical_offset_bytes=logical_offset,
                size_bytes=size,
                source_tensors=[f"draft_head_fit:{key}"],
            )
        )
        logical_offset += size

    payload = manifest.as_payload()
    payload["draft_heads"] = [asdict(head) for head in draft_heads]
    payload["files"] = [asdict(file_entry) for file_entry in files]
    payload["tensors"] = [asdict(entry) for entry in tensor_entries]
    (pack_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _collect_prefill_hidden_pairs(
    *,
    draft_runtime,
    verifier_runtime,
    chunks: list[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    draft_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for input_ids in chunks:
            input_ids = input_ids.to(draft_runtime.device)
            draft_hidden = run_semantic_stack_forward(draft_runtime, input_ids)
            target_hidden = run_semantic_stack_forward_from_hidden(verifier_runtime, draft_hidden)
            draft_rows.append(draft_hidden.squeeze(0).to(dtype=torch.float32).cpu())
            target_rows.append(target_hidden.squeeze(0).to(dtype=torch.float32).cpu())
    return torch.cat(draft_rows, dim=0), torch.cat(target_rows, dim=0)


def _collect_decode_hidden_pairs(
    *,
    draft_runtime,
    verifier_runtime,
    prompts: list[torch.LongTensor],
    prompt_tokens: int,
    decode_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive for decode calibration")
    draft_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    max_seq_len = prompt_tokens + decode_steps + 1
    position_cache = build_qwen_position_cache(
        draft_runtime.rope_inv_freq,
        draft_runtime.attention_scaling,
        max_seq_len=max_seq_len,
        dtype=draft_runtime.dtype,
    )
    with torch.inference_mode():
        for input_ids in prompts:
            input_ids = input_ids.to(draft_runtime.device)
            draft_hidden, draft_cache = run_semantic_stack_prefill(
                draft_runtime,
                input_ids,
                page_size=16,
                max_seq_len=max_seq_len,
                position_cache=position_cache,
            )
            verifier_hidden, verifier_cache = run_semantic_stack_prefill_from_hidden(
                verifier_runtime,
                draft_hidden,
                page_size=16,
                max_seq_len=max_seq_len,
                position_cache=position_cache,
            )
            exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden)
            for _ in range(decode_steps):
                draft_hidden, draft_cache = run_semantic_stack_decode(
                    draft_runtime,
                    exact_token,
                    draft_cache,
                    position_cache=position_cache,
                )
                verifier_hidden, verifier_cache = run_semantic_stack_decode_from_hidden(
                    verifier_runtime,
                    draft_hidden,
                    verifier_cache,
                    position_cache=position_cache,
                )
                draft_rows.append(draft_hidden.squeeze(0).to(dtype=torch.float32).cpu())
                target_rows.append(verifier_hidden.squeeze(0).to(dtype=torch.float32).cpu())
                exact_token = select_semantic_greedy_token(verifier_runtime, verifier_hidden)
    return torch.cat(draft_rows, dim=0), torch.cat(target_rows, dim=0)


def fit_qwen_draft_projection(
    *,
    model_path: str | Path,
    pack_dir: str | Path,
    draft_layer_count: int,
    key: str,
    chunk_tokens: int = 128,
    max_chunks: int = 32,
    ridge_lambda: float = 0.1,
    calibration_mode: str = "prefill",
    decode_steps: int = 16,
    device: str | torch.device = "cuda:0",
    dtype: str = "bfloat16",
    repo_root: str | Path | None = None,
) -> DraftHeadFitResult:
    pack_root = Path(pack_dir)
    repo_path = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    corpus_text = _repo_default_corpus(repo_path)

    draft_runtime = materialize_qwen_semantic_stack_from_leanpack(
        pack_root,
        layer_indices=tuple(range(draft_layer_count)),
        device=device,
        dtype=dtype,
        include_output_head=True,
    )
    verifier_runtime = materialize_qwen_semantic_stack_from_leanpack(
        pack_root,
        layer_indices=tuple(range(draft_layer_count, draft_runtime.config.num_hidden_layers)),
        device=device,
        dtype=dtype,
        include_output_head=True,
    )

    if calibration_mode == "prefill":
        chunks, corpus_tokens = _build_token_chunks(
            tokenizer,
            corpus_text,
            chunk_tokens=chunk_tokens,
            max_chunks=max_chunks,
        )
        x, y = _collect_prefill_hidden_pairs(
            draft_runtime=draft_runtime,
            verifier_runtime=verifier_runtime,
            chunks=chunks,
        )
        chunks_used = len(chunks)
        effective_decode_steps = 0
    elif calibration_mode == "decode":
        prompts, corpus_tokens = _build_prompt_chunks(
            tokenizer,
            corpus_text,
            prompt_tokens=chunk_tokens,
            max_prompts=max_chunks,
        )
        x, y = _collect_decode_hidden_pairs(
            draft_runtime=draft_runtime,
            verifier_runtime=verifier_runtime,
            prompts=prompts,
            prompt_tokens=chunk_tokens,
            decode_steps=decode_steps,
        )
        chunks_used = len(prompts)
        effective_decode_steps = decode_steps
    else:
        raise ValueError(f"unsupported calibration_mode: {calibration_mode}")

    projection_weight = _fit_linear_projection(x, y, ridge_lambda=ridge_lambda)
    projected = torch.matmul(x, projection_weight.transpose(0, 1))
    fit_result = DraftHeadFitResult(
        key=key,
        draft_layer_count=draft_layer_count,
        calibration_mode=calibration_mode,
        fit_samples=int(x.shape[0]),
        chunk_tokens=chunk_tokens,
        chunks_used=chunks_used,
        decode_steps=effective_decode_steps,
        ridge_lambda=float(ridge_lambda),
        projection_shape=(int(projection_weight.shape[0]), int(projection_weight.shape[1])),
        corpus_tokens=corpus_tokens,
        calibration_hidden_mse_before=float(torch.mean((x - y).pow(2)).item()),
        calibration_hidden_mse_after=float(torch.mean((projected - y).pow(2)).item()),
    )
    _update_manifest_with_draft_head(
        pack_dir=pack_root,
        key=key,
        draft_layer_count=draft_layer_count,
        projection_weight=projection_weight.to(dtype=torch.bfloat16),
        fit_result=fit_result,
    )
    return fit_result
