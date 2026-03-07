from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding


@dataclass(frozen=True)
class QwenWeightIndex:
    model_path: Path
    weight_map: dict[str, str]

    @classmethod
    def load(cls, model_path: str | Path) -> "QwenWeightIndex":
        root = Path(model_path)
        payload = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        return cls(model_path=root, weight_map=payload["weight_map"])

    def shard_for(self, tensor_name: str) -> Path:
        return self.model_path / self.weight_map[tensor_name]


@dataclass
class QwenBlockRuntime:
    model_path: Path
    config: Qwen3Config
    embed_tokens: torch.nn.Embedding
    layer: Qwen3DecoderLayer
    rotary: Qwen3RotaryEmbedding
    device: torch.device
    dtype: torch.dtype


@dataclass
class QwenStackRuntime:
    model_path: Path
    config: Qwen3Config
    layer_indices: tuple[int, ...]
    embed_tokens: torch.nn.Embedding
    layers: tuple[Qwen3DecoderLayer, ...]
    rotary: Qwen3RotaryEmbedding
    final_norm: Qwen3RMSNorm | None
    lm_head: torch.nn.Linear | None
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class QwenGenerationState:
    generated_ids: torch.LongTensor
    stop_reason: str
    cache_seq_length: int


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_qwen_config(model_path: str | Path) -> Qwen3Config:
    config = Qwen3Config.from_pretrained(model_path)
    if getattr(config, "_attn_implementation", None) in (None, ""):
        config._attn_implementation = "sdpa"
    return config


def stage_tensors_to_cpu(model_path: str | Path, tensor_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    index = QwenWeightIndex.load(model_path)
    grouped: dict[Path, list[str]] = {}
    for tensor_name in tensor_names:
        grouped.setdefault(index.shard_for(tensor_name), []).append(tensor_name)

    staged: dict[str, torch.Tensor] = {}
    for shard_path, shard_tensors in grouped.items():
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for tensor_name in shard_tensors:
                staged[tensor_name] = handle.get_tensor(tensor_name).contiguous()
    return staged


def qwen_block_tensor_names(layer_idx: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_idx}."
    return (
        "model.embed_tokens.weight",
        *qwen_layer_tensor_names(layer_idx),
    )


def qwen_layer_tensor_names(layer_idx: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_idx}."
    return (
        f"{prefix}input_layernorm.weight",
        f"{prefix}self_attn.q_proj.weight",
        f"{prefix}self_attn.k_proj.weight",
        f"{prefix}self_attn.v_proj.weight",
        f"{prefix}self_attn.o_proj.weight",
        f"{prefix}self_attn.q_norm.weight",
        f"{prefix}self_attn.k_norm.weight",
        f"{prefix}post_attention_layernorm.weight",
        f"{prefix}mlp.gate_proj.weight",
        f"{prefix}mlp.up_proj.weight",
        f"{prefix}mlp.down_proj.weight",
    )


def qwen_output_tensor_names() -> tuple[str, ...]:
    return ("model.norm.weight", "lm_head.weight")


def resolve_layer_indices(
    config: Qwen3Config,
    layer_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if not layer_indices:
        return tuple(range(config.num_hidden_layers))
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= config.num_hidden_layers:
            raise ValueError(
                f"layer index {layer_idx} out of range for num_hidden_layers={config.num_hidden_layers}"
            )
    return layer_indices


def _materialize_embed_tokens(
    model_path: str | Path,
    config: Qwen3Config,
    target_device: torch.device,
    torch_dtype: torch.dtype,
) -> torch.nn.Embedding:
    embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    staged = stage_tensors_to_cpu(model_path, ("model.embed_tokens.weight",))
    embed_tokens.weight.data.copy_(staged["model.embed_tokens.weight"].to(embed_tokens.weight.dtype))
    return embed_tokens.to(device=target_device, dtype=torch_dtype)


def _materialize_qwen_layer(
    model_path: str | Path,
    config: Qwen3Config,
    layer_idx: int,
    target_device: torch.device,
    torch_dtype: torch.dtype,
) -> Qwen3DecoderLayer:
    layer = Qwen3DecoderLayer(config, layer_idx)
    prefix = f"model.layers.{layer_idx}."
    layer_state = {
        tensor_name.removeprefix(prefix): tensor.to(torch.float32)
        for tensor_name, tensor in stage_tensors_to_cpu(model_path, qwen_layer_tensor_names(layer_idx)).items()
        if tensor_name.startswith(prefix)
    }
    layer.load_state_dict(layer_state, strict=True)
    return layer.to(device=target_device, dtype=torch_dtype)


def _materialize_qwen_output_head(
    model_path: str | Path,
    config: Qwen3Config,
    target_device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[Qwen3RMSNorm, torch.nn.Linear]:
    final_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    staged = stage_tensors_to_cpu(model_path, qwen_output_tensor_names())
    final_norm.weight.data.copy_(staged["model.norm.weight"].to(final_norm.weight.dtype))
    lm_head.weight.data.copy_(staged["lm_head.weight"].to(lm_head.weight.dtype))
    return (
        final_norm.to(device=target_device, dtype=torch_dtype),
        lm_head.to(device=target_device, dtype=torch_dtype),
    )


def materialize_qwen_block_runtime(
    model_path: str | Path,
    layer_idx: int,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
) -> QwenBlockRuntime:
    config = load_qwen_config(model_path)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)

    embed_tokens = _materialize_embed_tokens(model_path, config, target_device, torch_dtype)
    layer = _materialize_qwen_layer(model_path, config, layer_idx, target_device, torch_dtype)
    rotary = Qwen3RotaryEmbedding(config)
    rotary = rotary.to(device=target_device)

    return QwenBlockRuntime(
        model_path=Path(model_path),
        config=config,
        embed_tokens=embed_tokens,
        layer=layer,
        rotary=rotary,
        device=target_device,
        dtype=torch_dtype,
    )


def materialize_qwen_stack_runtime(
    model_path: str | Path,
    layer_indices: tuple[int, ...] | None = None,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = False,
) -> QwenStackRuntime:
    config = load_qwen_config(model_path)
    resolved_layer_indices = resolve_layer_indices(config, layer_indices)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)
    embed_tokens = _materialize_embed_tokens(model_path, config, target_device, torch_dtype)
    rotary = Qwen3RotaryEmbedding(config).to(device=target_device)
    layers = tuple(
        _materialize_qwen_layer(model_path, config, layer_idx, target_device, torch_dtype)
        for layer_idx in resolved_layer_indices
    )

    final_norm: Qwen3RMSNorm | None = None
    lm_head: torch.nn.Linear | None = None
    if include_output_head:
        final_norm, lm_head = _materialize_qwen_output_head(model_path, config, target_device, torch_dtype)

    return QwenStackRuntime(
        model_path=Path(model_path),
        config=config,
        layer_indices=resolved_layer_indices,
        embed_tokens=embed_tokens,
        layers=layers,
        rotary=rotary,
        final_norm=final_norm,
        lm_head=lm_head,
        device=target_device,
        dtype=torch_dtype,
    )


def materialize_qwen_full_runtime(
    model_path: str | Path,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = True,
) -> QwenStackRuntime:
    return materialize_qwen_stack_runtime(
        model_path=model_path,
        layer_indices=None,
        device=device,
        dtype=dtype,
        include_output_head=include_output_head,
    )


def build_prefill_attention_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def build_decode_attention_mask(
    query_len: int,
    total_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    past_len = total_len - query_len
    mask = torch.zeros((query_len, total_len), device=device, dtype=dtype)
    query_positions = torch.arange(query_len, device=device).unsqueeze(1)
    key_positions = torch.arange(total_len, device=device).unsqueeze(0)
    allowed = key_positions <= (past_len + query_positions)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def _extract_hidden_states(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"unsupported decoder-layer output type: {type(output)!r}")


def _run_qwen_layers(
    layers: tuple[Qwen3DecoderLayer, ...],
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    cache: DynamicCache | None,
    use_cache: bool,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    for layer in layers:
        output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = _extract_hidden_states(output)
    return hidden_states


def run_block_forward(runtime: QwenBlockRuntime, input_ids: torch.LongTensor) -> torch.Tensor:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=runtime.device).unsqueeze(0)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_prefill_attention_mask(seq_len, runtime.device, runtime.dtype)
    output = runtime.layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    return _extract_hidden_states(output)


def run_stack_forward(runtime: QwenStackRuntime, input_ids: torch.LongTensor) -> torch.Tensor:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=runtime.device).unsqueeze(0)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_prefill_attention_mask(seq_len, runtime.device, runtime.dtype)
    return _run_qwen_layers(
        runtime.layers,
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=position_embeddings,
    )


def run_single_layer_prefill(
    runtime: QwenBlockRuntime,
    input_ids: torch.LongTensor,
) -> tuple[torch.Tensor, DynamicCache]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=runtime.device).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=runtime.device)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_prefill_attention_mask(seq_len, runtime.device, runtime.dtype)
    cache = DynamicCache(config=runtime.config)
    output = runtime.layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    return _extract_hidden_states(output), cache


def run_stack_prefill(
    runtime: QwenStackRuntime,
    input_ids: torch.LongTensor,
) -> tuple[torch.Tensor, DynamicCache]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=runtime.device).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=runtime.device)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_prefill_attention_mask(seq_len, runtime.device, runtime.dtype)
    cache = DynamicCache(config=runtime.config)
    hidden_states = _run_qwen_layers(
        runtime.layers,
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    return hidden_states, cache


def run_single_layer_decode(
    runtime: QwenBlockRuntime,
    input_ids: torch.LongTensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    query_len = hidden_states.shape[1]
    past_len = cache.get_seq_length()
    total_len = past_len + query_len
    position_ids = torch.arange(past_len, total_len, device=runtime.device).unsqueeze(0)
    cache_position = torch.arange(past_len, total_len, device=runtime.device)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_decode_attention_mask(query_len, total_len, runtime.device, runtime.dtype)
    output = runtime.layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    return _extract_hidden_states(output), cache


def run_stack_decode(
    runtime: QwenStackRuntime,
    input_ids: torch.LongTensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = runtime.embed_tokens(input_ids)
    query_len = hidden_states.shape[1]
    past_len = cache.get_seq_length()
    total_len = past_len + query_len
    position_ids = torch.arange(past_len, total_len, device=runtime.device).unsqueeze(0)
    cache_position = torch.arange(past_len, total_len, device=runtime.device)
    position_embeddings = runtime.rotary(hidden_states, position_ids)
    attention_mask = build_decode_attention_mask(query_len, total_len, runtime.device, runtime.dtype)
    hidden_states = _run_qwen_layers(
        runtime.layers,
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    return hidden_states, cache


def project_hidden_to_logits(runtime: QwenStackRuntime, hidden_states: torch.Tensor) -> torch.Tensor:
    if runtime.final_norm is None or runtime.lm_head is None:
        raise ValueError("runtime does not include model.norm and lm_head")
    hidden_states = runtime.final_norm(hidden_states)
    return runtime.lm_head(hidden_states[:, -1:, :])


def normalize_stop_token_ids(stop_token_ids: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if stop_token_ids is None:
        return ()
    if isinstance(stop_token_ids, int):
        return (int(stop_token_ids),)
    return tuple(int(token_id) for token_id in stop_token_ids)


def select_greedy_token(logits: torch.Tensor) -> torch.LongTensor:
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def run_greedy_generation(
    runtime: QwenStackRuntime,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: int | tuple[int, ...] | list[int] | None = None,
    include_last_token_in_cache: bool = True,
) -> QwenGenerationState:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return QwenGenerationState(
            generated_ids=torch.empty((1, 0), device=runtime.device, dtype=torch.long),
            stop_reason="max_new_tokens",
            cache_seq_length=int(input_ids.shape[-1]),
        )

    stop_tokens = set(normalize_stop_token_ids(stop_token_ids))
    prefill_hidden, cache = run_stack_prefill(runtime, input_ids)
    next_logits = project_hidden_to_logits(runtime, prefill_hidden)
    generated: list[int] = []
    stop_reason = "max_new_tokens"

    for step in range(max_new_tokens):
        next_token = select_greedy_token(next_logits)
        token_id = int(next_token.item())
        generated.append(token_id)
        is_last_step = step == max_new_tokens - 1
        should_stop = token_id in stop_tokens or is_last_step

        if should_stop:
            if token_id in stop_tokens:
                stop_reason = "stop_token"
            if include_last_token_in_cache:
                _, cache = run_stack_decode(runtime, next_token, cache)
            break

        decode_hidden, cache = run_stack_decode(runtime, next_token, cache)
        next_logits = project_hidden_to_logits(runtime, decode_hidden)

    generated_ids = torch.tensor([generated], device=runtime.device, dtype=torch.long)
    return QwenGenerationState(
        generated_ids=generated_ids,
        stop_reason=stop_reason,
        cache_seq_length=int(cache.get_seq_length()),
    )
