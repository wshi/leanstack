from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

from .kv_cache import KVBlockManager, KVPageLayout


@dataclass(frozen=True)
class QwenWeightIndex:
    model_path: Path
    weight_map: dict[str, str]

    @classmethod
    def load(cls, model_path: str | Path) -> "QwenWeightIndex":
        root = Path(model_path)
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return cls(model_path=root, weight_map=payload["weight_map"])

        single_shard_path = root / "model.safetensors"
        if not single_shard_path.exists():
            raise FileNotFoundError(f"could not find {index_path.name} or {single_shard_path.name} under {root}")

        with safe_open(single_shard_path, framework="pt", device="cpu") as handle:
            weight_map = {tensor_name: single_shard_path.name for tensor_name in handle.keys()}
        return cls(model_path=root, weight_map=weight_map)

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


@dataclass
class QwenSemanticAttentionRuntime:
    layer_idx: int
    q_proj_weight: torch.Tensor
    k_proj_weight: torch.Tensor
    v_proj_weight: torch.Tensor
    qkv_proj_weight: torch.Tensor
    q_proj_out_dim: int
    k_proj_out_dim: int
    o_proj_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor


@dataclass
class QwenSemanticMlpRuntime:
    gate_proj_weight: torch.Tensor
    up_proj_weight: torch.Tensor
    gate_up_proj_weight: torch.Tensor
    gate_proj_out_dim: int
    down_proj_weight: torch.Tensor


@dataclass
class QwenSemanticBlockRuntime:
    model_path: Path
    config: Qwen3Config
    layer_idx: int
    embed_tokens_weight: torch.Tensor
    input_layernorm_weight: torch.Tensor
    attention: QwenSemanticAttentionRuntime
    post_attention_layernorm_weight: torch.Tensor
    mlp: QwenSemanticMlpRuntime
    rope_inv_freq: torch.Tensor
    attention_scaling: float
    device: torch.device
    dtype: torch.dtype


@dataclass
class QwenSemanticLayerRuntime:
    layer_idx: int
    input_layernorm_weight: torch.Tensor
    attention: QwenSemanticAttentionRuntime
    post_attention_layernorm_weight: torch.Tensor
    mlp: QwenSemanticMlpRuntime


@dataclass
class QwenSemanticStackRuntime:
    model_path: Path
    config: Qwen3Config
    layer_indices: tuple[int, ...]
    embed_tokens_weight: torch.Tensor
    layers: tuple[QwenSemanticLayerRuntime, ...]
    rope_inv_freq: torch.Tensor
    attention_scaling: float
    final_norm_weight: torch.Tensor | None
    lm_head_weight: torch.Tensor | None
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


def qwen_has_separate_lm_head(model_path: str | Path) -> bool:
    return "lm_head.weight" in QwenWeightIndex.load(model_path).weight_map


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
    lm_head_tensor_name = "lm_head.weight" if qwen_has_separate_lm_head(model_path) else "model.embed_tokens.weight"
    staged = stage_tensors_to_cpu(model_path, ("model.norm.weight", lm_head_tensor_name))
    final_norm.weight.data.copy_(staged["model.norm.weight"].to(final_norm.weight.dtype))
    lm_head.weight.data.copy_(staged[lm_head_tensor_name].to(lm_head.weight.dtype))
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


def materialize_qwen_semantic_block_runtime(
    model_path: str | Path,
    layer_idx: int,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
) -> QwenSemanticBlockRuntime:
    config = load_qwen_config(model_path)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rope_theta = config.rope_parameters["rope_theta"]
    rope_inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=target_device, dtype=torch.float32) / head_dim)
    )

    staged = stage_tensors_to_cpu(model_path, qwen_block_tensor_names(layer_idx))
    prefix = f"model.layers.{layer_idx}."

    def staged_weight(name: str) -> torch.Tensor:
        return staged[name].to(device=target_device, dtype=torch_dtype).contiguous()

    q_proj_weight = staged_weight(f"{prefix}self_attn.q_proj.weight")
    k_proj_weight = staged_weight(f"{prefix}self_attn.k_proj.weight")
    v_proj_weight = staged_weight(f"{prefix}self_attn.v_proj.weight")
    gate_proj_weight = staged_weight(f"{prefix}mlp.gate_proj.weight")
    up_proj_weight = staged_weight(f"{prefix}mlp.up_proj.weight")

    return QwenSemanticBlockRuntime(
        model_path=Path(model_path),
        config=config,
        layer_idx=layer_idx,
        embed_tokens_weight=staged_weight("model.embed_tokens.weight"),
        input_layernorm_weight=staged_weight(f"{prefix}input_layernorm.weight"),
        attention=QwenSemanticAttentionRuntime(
            layer_idx=layer_idx,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            qkv_proj_weight=torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous(),
            q_proj_out_dim=staged[f"{prefix}self_attn.q_proj.weight"].shape[0],
            k_proj_out_dim=staged[f"{prefix}self_attn.k_proj.weight"].shape[0],
            o_proj_weight=staged_weight(f"{prefix}self_attn.o_proj.weight"),
            q_norm_weight=staged_weight(f"{prefix}self_attn.q_norm.weight"),
            k_norm_weight=staged_weight(f"{prefix}self_attn.k_norm.weight"),
        ),
        post_attention_layernorm_weight=staged_weight(f"{prefix}post_attention_layernorm.weight"),
        mlp=QwenSemanticMlpRuntime(
            gate_proj_weight=gate_proj_weight,
            up_proj_weight=up_proj_weight,
            gate_up_proj_weight=torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous(),
            gate_proj_out_dim=staged[f"{prefix}mlp.gate_proj.weight"].shape[0],
            down_proj_weight=staged_weight(f"{prefix}mlp.down_proj.weight"),
        ),
        rope_inv_freq=rope_inv_freq,
        attention_scaling=1.0,
        device=target_device,
        dtype=torch_dtype,
    )


def _materialize_qwen_semantic_layer(
    model_path: str | Path,
    layer_idx: int,
    target_device: torch.device,
    torch_dtype: torch.dtype,
) -> QwenSemanticLayerRuntime:
    staged = stage_tensors_to_cpu(model_path, qwen_layer_tensor_names(layer_idx))
    prefix = f"model.layers.{layer_idx}."

    def staged_weight(name: str) -> torch.Tensor:
        return staged[name].to(device=target_device, dtype=torch_dtype).contiguous()

    q_proj_weight = staged_weight(f"{prefix}self_attn.q_proj.weight")
    k_proj_weight = staged_weight(f"{prefix}self_attn.k_proj.weight")
    v_proj_weight = staged_weight(f"{prefix}self_attn.v_proj.weight")
    gate_proj_weight = staged_weight(f"{prefix}mlp.gate_proj.weight")
    up_proj_weight = staged_weight(f"{prefix}mlp.up_proj.weight")

    return QwenSemanticLayerRuntime(
        layer_idx=layer_idx,
        input_layernorm_weight=staged_weight(f"{prefix}input_layernorm.weight"),
        attention=QwenSemanticAttentionRuntime(
            layer_idx=layer_idx,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            qkv_proj_weight=torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous(),
            q_proj_out_dim=staged[f"{prefix}self_attn.q_proj.weight"].shape[0],
            k_proj_out_dim=staged[f"{prefix}self_attn.k_proj.weight"].shape[0],
            o_proj_weight=staged_weight(f"{prefix}self_attn.o_proj.weight"),
            q_norm_weight=staged_weight(f"{prefix}self_attn.q_norm.weight"),
            k_norm_weight=staged_weight(f"{prefix}self_attn.k_norm.weight"),
        ),
        post_attention_layernorm_weight=staged_weight(f"{prefix}post_attention_layernorm.weight"),
        mlp=QwenSemanticMlpRuntime(
            gate_proj_weight=gate_proj_weight,
            up_proj_weight=up_proj_weight,
            gate_up_proj_weight=torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous(),
            gate_proj_out_dim=staged[f"{prefix}mlp.gate_proj.weight"].shape[0],
            down_proj_weight=staged_weight(f"{prefix}mlp.down_proj.weight"),
        ),
    )


def materialize_qwen_semantic_stack_runtime(
    model_path: str | Path,
    layer_indices: tuple[int, ...] | None = None,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = False,
) -> QwenSemanticStackRuntime:
    config = load_qwen_config(model_path)
    resolved_layer_indices = resolve_layer_indices(config, layer_indices)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rope_theta = config.rope_parameters["rope_theta"]
    rope_inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=target_device, dtype=torch.float32) / head_dim)
    )

    embed_tokens_weight = stage_tensors_to_cpu(model_path, ("model.embed_tokens.weight",))[
        "model.embed_tokens.weight"
    ].to(device=target_device, dtype=torch_dtype)
    layers = tuple(
        _materialize_qwen_semantic_layer(model_path, layer_idx, target_device, torch_dtype)
        for layer_idx in resolved_layer_indices
    )

    final_norm_weight: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    if include_output_head:
        final_norm_weight = stage_tensors_to_cpu(model_path, ("model.norm.weight",))["model.norm.weight"].to(
            device=target_device,
            dtype=torch_dtype,
        ).contiguous()
        if qwen_has_separate_lm_head(model_path):
            lm_head_weight = stage_tensors_to_cpu(model_path, ("lm_head.weight",))["lm_head.weight"].to(
                device=target_device,
                dtype=torch_dtype,
            ).contiguous()
        else:
            lm_head_weight = embed_tokens_weight

    return QwenSemanticStackRuntime(
        model_path=Path(model_path),
        config=config,
        layer_indices=resolved_layer_indices,
        embed_tokens_weight=embed_tokens_weight,
        layers=layers,
        rope_inv_freq=rope_inv_freq,
        attention_scaling=1.0,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        device=target_device,
        dtype=torch_dtype,
    )


def materialize_qwen_full_semantic_runtime(
    model_path: str | Path,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = True,
) -> QwenSemanticStackRuntime:
    return materialize_qwen_semantic_stack_runtime(
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


def _compile_reduce_overhead(fn: Callable[..., Any]) -> Callable[..., Any]:
    if not hasattr(torch, "compile"):
        return fn

    compiled_fn: Callable[..., Any] | None = None
    compile_enabled = True

    def wrapper(*args, **kwargs):
        nonlocal compiled_fn, compile_enabled
        if not compile_enabled:
            return fn(*args, **kwargs)
        if compiled_fn is None:
            try:
                compiled_fn = torch.compile(fn, mode="default", options={"triton.cudagraphs": False})
            except TypeError:
                compiled_fn = torch.compile(fn, mode="default")
            except Exception:
                compile_enabled = False
                return fn(*args, **kwargs)
        try:
            return compiled_fn(*args, **kwargs)
        except Exception:
            compile_enabled = False
            compiled_fn = None
            return fn(*args, **kwargs)

    return wrapper


def qwen_rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    try:
        return F.rms_norm(
            hidden_states,
            normalized_shape=(hidden_states.shape[-1],),
            weight=weight,
            eps=eps,
        )
    except AttributeError:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)
        return weight * normalized.to(input_dtype)


def qwen_rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def build_qwen_position_embeddings(
    rope_inv_freq: torch.Tensor,
    attention_scaling: float,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq_expanded = rope_inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    cos = freqs.cos() * attention_scaling
    sin = freqs.sin() * attention_scaling
    return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


def build_qwen_position_cache(
    rope_inv_freq: torch.Tensor,
    attention_scaling: float,
    max_seq_len: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    position_ids = torch.arange(max_seq_len, device=rope_inv_freq.device, dtype=torch.float32)
    freqs = torch.outer(position_ids, rope_inv_freq.float())
    cos = (freqs.cos() * attention_scaling).to(dtype=dtype)
    sin = (freqs.sin() * attention_scaling).to(dtype=dtype)
    return cos, sin


def lookup_qwen_position_embeddings(
    position_cache: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_cache, sin_cache = position_cache
    flat_position_ids = position_ids.reshape(-1)
    cos = cos_cache.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    sin = sin_cache.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    return cos, sin


def qwen_apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_first_half, query_second_half = query_states.chunk(2, dim=-1)
    key_first_half, key_second_half = key_states.chunk(2, dim=-1)
    return (
        torch.cat(
            (
                (query_first_half * cos) - (query_second_half * sin),
                (query_second_half * cos) + (query_first_half * sin),
            ),
            dim=-1,
        ),
        torch.cat(
            (
                (key_first_half * cos) - (key_second_half * sin),
                (key_second_half * cos) + (key_first_half * sin),
            ),
            dim=-1,
        ),
    )


def slice_qwen_position_embeddings(
    position_cache: tuple[torch.Tensor, torch.Tensor],
    start: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length <= 0:
        raise ValueError("length must be positive")
    cos_cache, sin_cache = position_cache
    return (
        cos_cache.narrow(0, start, length).unsqueeze(0),
        sin_cache.narrow(0, start, length).unsqueeze(0),
    )


def _semantic_attention_qkv_eager(
    hidden_states: torch.Tensor,
    qkv_proj_weight: torch.Tensor,
    q_proj_out_dim: int,
    k_proj_out_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_len, _ = hidden_states.shape
    qkv_states = F.linear(hidden_states, qkv_proj_weight)
    query_states, key_states, value_states = torch.split(
        qkv_states,
        (q_proj_out_dim, k_proj_out_dim, k_proj_out_dim),
        dim=-1,
    )
    query_states = query_states.view(batch_size, query_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, query_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, query_len, num_key_value_heads, head_dim)
    query_states = F.rms_norm(
        query_states,
        normalized_shape=(head_dim,),
        weight=q_norm_weight,
        eps=rms_norm_eps,
    ).transpose(1, 2)
    key_states = F.rms_norm(
        key_states,
        normalized_shape=(head_dim,),
        weight=k_norm_weight,
        eps=rms_norm_eps,
    ).transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    return qwen_apply_rotary_pos_emb(query_states, key_states, cos, sin) + (value_states,)


def _semantic_post_attention_mlp_eager(
    attention_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    gate_proj_out_dim: int,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
) -> torch.Tensor:
    hidden_states = F.linear(attention_output, o_proj_weight)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = F.rms_norm(
        hidden_states,
        normalized_shape=(hidden_states.shape[-1],),
        weight=post_attention_layernorm_weight,
        eps=rms_norm_eps,
    )
    gate_up = F.linear(hidden_states, gate_up_proj_weight)
    gate_states, up_states = torch.split(gate_up, (gate_proj_out_dim, gate_up.shape[-1] - gate_proj_out_dim), dim=-1)
    hidden_states = F.linear(F.silu(gate_states) * up_states, down_proj_weight)
    return residual + hidden_states


def _semantic_greedy_head_eager(
    hidden_states: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    rms_norm_eps: float,
) -> torch.LongTensor:
    hidden_states = F.rms_norm(
        hidden_states,
        normalized_shape=(hidden_states.shape[-1],),
        weight=final_norm_weight,
        eps=rms_norm_eps,
    )
    logits = F.linear(hidden_states[:, -1:, :], lm_head_weight)
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


_semantic_attention_qkv = _compile_reduce_overhead(_semantic_attention_qkv_eager)
_semantic_post_attention_mlp = _compile_reduce_overhead(_semantic_post_attention_mlp_eager)
_semantic_greedy_head = _compile_reduce_overhead(_semantic_greedy_head_eager)


def qwen_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def eager_qwen_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    repeated_keys = qwen_repeat_kv(key_states, num_key_value_groups)
    repeated_values = qwen_repeat_kv(value_states, num_key_value_groups)
    attention_scores = torch.matmul(query_states, repeated_keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    attention_scores = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attention_output = torch.matmul(attention_scores, repeated_values)
    return attention_output.transpose(1, 2).contiguous()


def sdpa_qwen_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    is_decode = query_states.shape[-2] == 1 and key_states.shape[-2] >= 1
    use_causal = not is_decode and query_states.shape[-2] == key_states.shape[-2]
    sdpa_kwargs = {
        "attn_mask": None if use_causal or is_decode else attention_mask,
        "dropout_p": 0.0,
        "is_causal": use_causal,
        "scale": scaling,
    }
    if num_key_value_groups > 1:
        try:
            attention_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                enable_gqa=True,
                **sdpa_kwargs,
            )
        except TypeError:
            attention_output = F.scaled_dot_product_attention(
                query_states,
                qwen_repeat_kv(key_states, num_key_value_groups),
                qwen_repeat_kv(value_states, num_key_value_groups),
                **sdpa_kwargs,
            )
    else:
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            **sdpa_kwargs,
        )
    return attention_output.transpose(1, 2).contiguous()


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


def build_qwen_kv_cache(
    config: Qwen3Config,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    page_size: int = 16,
    batch_size: int = 1,
) -> KVBlockManager:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    layout = KVPageLayout(
        num_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    return KVBlockManager(layout)


def _semantic_qwen_attention_forward(
    config: Qwen3Config,
    attention: QwenSemanticAttentionRuntime,
    rope_inv_freq: torch.Tensor,
    attention_scaling: float,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
    kv_cache: KVBlockManager | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    if position_embeddings is None:
        if position_ids is None:
            raise ValueError("position_ids are required when position_embeddings are not provided")
        cos, sin = build_qwen_position_embeddings(
            rope_inv_freq,
            attention_scaling,
            hidden_states,
            position_ids,
        )
    else:
        cos, sin = position_embeddings
    query_states, key_states, value_states = _semantic_attention_qkv(
        hidden_states,
        attention.qkv_proj_weight,
        attention.q_proj_out_dim,
        attention.k_proj_out_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
        attention.q_norm_weight,
        attention.k_norm_weight,
        cos,
        sin,
        config.rms_norm_eps,
    )

    if kv_cache is not None:
        kv_cache.append(attention.layer_idx, key_states, value_states)
        key_states, value_states = kv_cache.get(attention.layer_idx)

    attention_output = sdpa_qwen_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=head_dim**-0.5,
        num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
    )
    return attention_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)


def semantic_qwen_attention_forward(
    runtime: QwenSemanticBlockRuntime,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
    kv_cache: KVBlockManager | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    return _semantic_qwen_attention_forward(
        config=runtime.config,
        attention=runtime.attention,
        rope_inv_freq=runtime.rope_inv_freq,
        attention_scaling=runtime.attention_scaling,
        hidden_states=hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        position_embeddings=position_embeddings,
    )


def semantic_qwen_mlp_forward(mlp: QwenSemanticMlpRuntime, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(hidden_states, mlp.gate_up_proj_weight)
    gate_states, up_states = torch.split(
        gate_up,
        (mlp.gate_proj_out_dim, gate_up.shape[-1] - mlp.gate_proj_out_dim),
        dim=-1,
    )
    return F.linear(F.silu(gate_states) * up_states, mlp.down_proj_weight)


def semantic_qwen_layer_forward(
    config: Qwen3Config,
    layer: QwenSemanticLayerRuntime,
    rope_inv_freq: torch.Tensor,
    attention_scaling: float,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
    kv_cache: KVBlockManager | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = qwen_rmsnorm(hidden_states, layer.input_layernorm_weight, config.rms_norm_eps)
    attention_output = _semantic_qwen_attention_forward(
        config=config,
        attention=layer.attention,
        rope_inv_freq=rope_inv_freq,
        attention_scaling=attention_scaling,
        hidden_states=hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        position_embeddings=position_embeddings,
    )
    return _semantic_post_attention_mlp(
        attention_output,
        residual,
        layer.attention.o_proj_weight,
        layer.post_attention_layernorm_weight,
        layer.mlp.gate_up_proj_weight,
        layer.mlp.gate_proj_out_dim,
        layer.mlp.down_proj_weight,
        config.rms_norm_eps,
    )


def run_semantic_qwen_block(
    runtime: QwenSemanticBlockRuntime,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
    kv_cache: KVBlockManager | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = qwen_rmsnorm(hidden_states, runtime.input_layernorm_weight, runtime.config.rms_norm_eps)
    hidden_states = semantic_qwen_attention_forward(
        runtime,
        hidden_states,
        position_ids,
        attention_mask,
        kv_cache,
        position_embeddings=position_embeddings,
    )
    return _semantic_post_attention_mlp(
        hidden_states,
        residual,
        runtime.attention.o_proj_weight,
        runtime.post_attention_layernorm_weight,
        runtime.mlp.gate_up_proj_weight,
        runtime.mlp.gate_proj_out_dim,
        runtime.mlp.down_proj_weight,
        runtime.config.rms_norm_eps,
    )


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


def run_semantic_block_forward(runtime: QwenSemanticBlockRuntime, input_ids: torch.LongTensor) -> torch.Tensor:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    seq_len = hidden_states.shape[1]
    position_cache = build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=seq_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(position_cache, start=0, length=seq_len)
    return run_semantic_qwen_block(
        runtime,
        hidden_states,
        None,
        attention_mask=None,
        kv_cache=None,
        position_embeddings=position_embeddings,
    )


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


def run_semantic_stack_forward(runtime: QwenSemanticStackRuntime, input_ids: torch.LongTensor) -> torch.Tensor:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    seq_len = hidden_states.shape[1]
    position_cache = build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=seq_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(position_cache, start=0, length=seq_len)
    for layer in runtime.layers:
        hidden_states = semantic_qwen_layer_forward(
            config=runtime.config,
            layer=layer,
            rope_inv_freq=runtime.rope_inv_freq,
            attention_scaling=runtime.attention_scaling,
            hidden_states=hidden_states,
            position_ids=None,
            attention_mask=None,
            kv_cache=None,
            position_embeddings=position_embeddings,
        )
    return hidden_states


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


def run_semantic_single_layer_prefill(
    runtime: QwenSemanticBlockRuntime,
    input_ids: torch.LongTensor,
    page_size: int = 16,
    max_seq_len: int | None = None,
    position_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, KVBlockManager]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    seq_len = hidden_states.shape[1]
    kv_cache = build_qwen_kv_cache(
        runtime.config,
        max_seq_len=max_seq_len or seq_len,
        device=runtime.device,
        dtype=runtime.dtype,
        page_size=page_size,
    )
    resolved_position_cache = position_cache or build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=max_seq_len or seq_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(resolved_position_cache, start=0, length=seq_len)
    hidden_states = run_semantic_qwen_block(
        runtime,
        hidden_states,
        None,
        attention_mask=None,
        kv_cache=kv_cache,
        position_embeddings=position_embeddings,
    )
    return hidden_states, kv_cache


def run_semantic_stack_prefill(
    runtime: QwenSemanticStackRuntime,
    input_ids: torch.LongTensor,
    page_size: int = 16,
    max_seq_len: int | None = None,
    position_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, KVBlockManager]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    seq_len = hidden_states.shape[1]
    kv_cache = build_qwen_kv_cache(
        runtime.config,
        max_seq_len=max_seq_len or seq_len,
        device=runtime.device,
        dtype=runtime.dtype,
        page_size=page_size,
    )
    resolved_position_cache = position_cache or build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=max_seq_len or seq_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(resolved_position_cache, start=0, length=seq_len)
    for layer in runtime.layers:
        hidden_states = semantic_qwen_layer_forward(
            config=runtime.config,
            layer=layer,
            rope_inv_freq=runtime.rope_inv_freq,
            attention_scaling=runtime.attention_scaling,
            hidden_states=hidden_states,
            position_ids=None,
            attention_mask=None,
            kv_cache=kv_cache,
            position_embeddings=position_embeddings,
        )
    return hidden_states, kv_cache


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


def run_semantic_single_layer_decode(
    runtime: QwenSemanticBlockRuntime,
    input_ids: torch.LongTensor,
    kv_cache: KVBlockManager,
    position_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, KVBlockManager]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    query_len = hidden_states.shape[1]
    past_len = kv_cache.get_seq_length()
    total_len = past_len + query_len
    resolved_position_cache = position_cache or build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=total_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(resolved_position_cache, start=past_len, length=query_len)
    hidden_states = run_semantic_qwen_block(
        runtime,
        hidden_states,
        None,
        attention_mask=None,
        kv_cache=kv_cache,
        position_embeddings=position_embeddings,
    )
    return hidden_states, kv_cache


def run_semantic_stack_decode(
    runtime: QwenSemanticStackRuntime,
    input_ids: torch.LongTensor,
    kv_cache: KVBlockManager,
    position_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, KVBlockManager]:
    input_ids = input_ids.to(runtime.device)
    hidden_states = F.embedding(input_ids, runtime.embed_tokens_weight)
    query_len = hidden_states.shape[1]
    past_len = kv_cache.get_seq_length()
    total_len = past_len + query_len
    resolved_position_cache = position_cache or build_qwen_position_cache(
        runtime.rope_inv_freq,
        runtime.attention_scaling,
        max_seq_len=total_len,
        dtype=runtime.dtype,
    )
    position_embeddings = slice_qwen_position_embeddings(resolved_position_cache, start=past_len, length=query_len)
    for layer in runtime.layers:
        hidden_states = semantic_qwen_layer_forward(
            config=runtime.config,
            layer=layer,
            rope_inv_freq=runtime.rope_inv_freq,
            attention_scaling=runtime.attention_scaling,
            hidden_states=hidden_states,
            position_ids=None,
            attention_mask=None,
            kv_cache=kv_cache,
            position_embeddings=position_embeddings,
        )
    return hidden_states, kv_cache


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


def project_semantic_hidden_to_logits(
    runtime: QwenSemanticStackRuntime,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if runtime.final_norm_weight is None or runtime.lm_head_weight is None:
        raise ValueError("runtime does not include model.norm and lm_head")
    hidden_states = qwen_rmsnorm(hidden_states, runtime.final_norm_weight, runtime.config.rms_norm_eps)
    return F.linear(hidden_states[:, -1:, :], runtime.lm_head_weight)


def select_semantic_greedy_token(
    runtime: QwenSemanticStackRuntime,
    hidden_states: torch.Tensor,
) -> torch.LongTensor:
    if runtime.final_norm_weight is None or runtime.lm_head_weight is None:
        raise ValueError("runtime does not include model.norm and lm_head")
    return _semantic_greedy_head(
        hidden_states,
        runtime.final_norm_weight,
        runtime.lm_head_weight,
        runtime.config.rms_norm_eps,
    )


def normalize_stop_token_ids(stop_token_ids: int | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if stop_token_ids is None:
        return ()
    if isinstance(stop_token_ids, int):
        return (int(stop_token_ids),)
    return tuple(int(token_id) for token_id in stop_token_ids)


def select_greedy_token(logits: torch.Tensor) -> torch.LongTensor:
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
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
    stop_token_tensor = (
        torch.tensor(sorted(stop_tokens), device=runtime.device, dtype=torch.long) if stop_tokens else None
    )
    prefill_hidden, cache = run_stack_prefill(runtime, input_ids)
    next_logits = project_hidden_to_logits(runtime, prefill_hidden)
    generated_gpu = torch.empty((max_new_tokens,), device=runtime.device, dtype=torch.long)
    emitted_tokens = 0
    stop_reason = "max_new_tokens"

    for step in range(max_new_tokens):
        next_token = select_greedy_token(next_logits)
        generated_gpu[step] = next_token.reshape(())
        emitted_tokens = step + 1
        is_last_step = step == max_new_tokens - 1
        should_stop = is_last_step
        stop_hit = False
        if stop_token_tensor is not None:
            stop_hit = bool(torch.eq(next_token.reshape(1), stop_token_tensor).any().item())
            should_stop = should_stop or stop_hit

        if should_stop:
            if stop_hit:
                stop_reason = "stop_token"
            if include_last_token_in_cache:
                _, cache = run_stack_decode(runtime, next_token, cache)
            break

        decode_hidden, cache = run_stack_decode(runtime, next_token, cache)
        next_logits = project_hidden_to_logits(runtime, decode_hidden)

    generated_ids = generated_gpu[:emitted_tokens].unsqueeze(0)
    return QwenGenerationState(
        generated_ids=generated_ids,
        stop_reason=stop_reason,
        cache_seq_length=int(cache.get_seq_length()),
    )
