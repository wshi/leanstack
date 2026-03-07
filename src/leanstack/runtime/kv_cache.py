from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVPageLayout:
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    page_size: int
    max_seq_len: int
    batch_size: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def num_pages(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size

    def pages_for_tokens(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return (token_count + self.page_size - 1) // self.page_size


class KVBlockManager:
    def __init__(self, layout: KVPageLayout):
        self.layout = layout
        shape = (
            layout.num_layers,
            layout.batch_size,
            layout.num_key_value_heads,
            layout.max_seq_len,
            layout.head_dim,
        )
        self.key_cache = torch.zeros(shape, device=layout.device, dtype=layout.dtype)
        self.value_cache = torch.zeros(shape, device=layout.device, dtype=layout.dtype)
        self.layer_seq_lens = torch.zeros(layout.num_layers, dtype=torch.int32)

    def append(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape != value_states.shape:
            raise ValueError("key/value states must have the same shape")

        batch_size, num_heads, token_count, head_dim = key_states.shape
        if batch_size != self.layout.batch_size:
            raise ValueError(f"expected batch_size={self.layout.batch_size}, got {batch_size}")
        if num_heads != self.layout.num_key_value_heads:
            raise ValueError(f"expected num_key_value_heads={self.layout.num_key_value_heads}, got {num_heads}")
        if head_dim != self.layout.head_dim:
            raise ValueError(f"expected head_dim={self.layout.head_dim}, got {head_dim}")

        start = int(self.layer_seq_lens[layer_idx].item())
        end = start + token_count
        if end > self.layout.max_seq_len:
            raise ValueError(
                f"cache overflow for layer {layer_idx}: requested {end} tokens with max_seq_len={self.layout.max_seq_len}"
            )

        self.key_cache[layer_idx, :, :, start:end, :].copy_(key_states)
        self.value_cache[layer_idx, :, :, start:end, :].copy_(value_states)
        self.layer_seq_lens[layer_idx] = end

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.layer_seq_lens[layer_idx].item())
        return (
            self.key_cache[layer_idx, :, :, :end, :],
            self.value_cache[layer_idx, :, :, :end, :],
        )

    def get_layer_seq_length(self, layer_idx: int) -> int:
        return int(self.layer_seq_lens[layer_idx].item())

    def get_seq_length(self) -> int:
        return int(self.layer_seq_lens.max().item())

    def page_table(self, layer_idx: int) -> tuple[int, ...]:
        token_count = self.get_layer_seq_length(layer_idx)
        return tuple(range(self.layout.pages_for_tokens(token_count)))

    def summary(self) -> dict[str, int]:
        seq_len = self.get_seq_length()
        return {
            "page_size": self.layout.page_size,
            "num_pages": self.layout.num_pages,
            "used_pages": self.layout.pages_for_tokens(seq_len),
            "seq_len": seq_len,
        }
