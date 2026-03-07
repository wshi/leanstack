from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


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


def materialize_qwen_block_runtime(
    model_path: str | Path,
    layer_idx: int,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
) -> QwenBlockRuntime:
    config = load_qwen_config(model_path)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)

    embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    layer = Qwen3DecoderLayer(config, layer_idx)
    rotary = Qwen3RotaryEmbedding(config)

    staged = stage_tensors_to_cpu(model_path, qwen_block_tensor_names(layer_idx))
    embed_tokens.weight.data.copy_(staged["model.embed_tokens.weight"].to(embed_tokens.weight.dtype))

    prefix = f"model.layers.{layer_idx}."
    layer_state = {
        tensor_name.removeprefix(prefix): tensor.to(torch.float32)
        for tensor_name, tensor in staged.items()
        if tensor_name.startswith(prefix)
    }
    layer.load_state_dict(layer_state, strict=True)

    embed_tokens = embed_tokens.to(device=target_device, dtype=torch_dtype)
    layer = layer.to(device=target_device, dtype=torch_dtype)
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
