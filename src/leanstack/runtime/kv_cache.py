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
            layout.num_pages,
            layout.page_size,
            layout.head_dim,
        )
        self.key_pages = torch.zeros(shape, device=layout.device, dtype=layout.dtype)
        self.value_pages = torch.zeros(shape, device=layout.device, dtype=layout.dtype)
        # Page-table metadata is host-managed. The runtime never consumes these scalars inside
        # GPU kernels, so keeping them on device only adds sync overhead through repeated .item().
        self.layer_page_table = [[-1 for _ in range(layout.num_pages)] for _ in range(layout.num_layers)]
        self.layer_page_counts = [0 for _ in range(layout.num_layers)]
        self.layer_seq_lens = [0 for _ in range(layout.num_layers)]

    def _ensure_page(self, layer_idx: int, logical_page_idx: int) -> int:
        physical_page_idx = self.layer_page_table[layer_idx][logical_page_idx]
        if physical_page_idx >= 0:
            return physical_page_idx

        next_page_idx = self.layer_page_counts[layer_idx]
        if next_page_idx >= self.layout.num_pages:
            raise ValueError(
                f"page allocation overflow for layer {layer_idx}: requested logical page {logical_page_idx} "
                f"with max pages={self.layout.num_pages}"
            )
        self.layer_page_table[layer_idx][logical_page_idx] = next_page_idx
        self.layer_page_counts[layer_idx] = next_page_idx + 1
        return next_page_idx

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

        start = self.layer_seq_lens[layer_idx]
        end = start + token_count
        if end > self.layout.max_seq_len:
            raise ValueError(
                f"cache overflow for layer {layer_idx}: requested {end} tokens with max_seq_len={self.layout.max_seq_len}"
            )

        if token_count == 1:
            logical_page_idx = start // self.layout.page_size
            page_offset = start % self.layout.page_size
            physical_page_idx = self._ensure_page(layer_idx, logical_page_idx)
            target_slice = slice(page_offset, page_offset + 1)
            self.key_pages[layer_idx, :, :, physical_page_idx, target_slice, :].copy_(key_states)
            self.value_pages[layer_idx, :, :, physical_page_idx, target_slice, :].copy_(value_states)
            self.layer_seq_lens[layer_idx] = end
            return

        source_offset = 0
        cursor = start
        while cursor < end:
            logical_page_idx = cursor // self.layout.page_size
            page_offset = cursor % self.layout.page_size
            copy_tokens = min(end - cursor, self.layout.page_size - page_offset)
            physical_page_idx = self._ensure_page(layer_idx, logical_page_idx)
            source_slice = slice(source_offset, source_offset + copy_tokens)
            target_slice = slice(page_offset, page_offset + copy_tokens)
            self.key_pages[layer_idx, :, :, physical_page_idx, target_slice, :].copy_(
                key_states[:, :, source_slice, :]
            )
            self.value_pages[layer_idx, :, :, physical_page_idx, target_slice, :].copy_(
                value_states[:, :, source_slice, :]
            )
            source_offset += copy_tokens
            cursor += copy_tokens

        self.layer_seq_lens[layer_idx] = end

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.get_layer_seq_length(layer_idx)
        if seq_len == 0:
            empty = torch.empty(
                (
                    self.layout.batch_size,
                    self.layout.num_key_value_heads,
                    0,
                    self.layout.head_dim,
                ),
                device=self.layout.device,
                dtype=self.layout.dtype,
            )
            return empty, empty.clone()

        page_count = self.layout.pages_for_tokens(seq_len)
        allocated_pages = self.layer_page_counts[layer_idx]
        if allocated_pages < page_count:
            raise ValueError(
                f"missing physical pages for layer {layer_idx}: have {allocated_pages}, need {page_count}"
            )

        # Pages are allocated monotonically for each layer, so the live prefix can be viewed
        # as a contiguous [tokens] axis without rebuilding it with torch.cat on every decode step.
        keys = self.key_pages[layer_idx, :, :, :page_count].reshape(
            self.layout.batch_size,
            self.layout.num_key_value_heads,
            page_count * self.layout.page_size,
            self.layout.head_dim,
        )[:, :, :seq_len, :]
        values = self.value_pages[layer_idx, :, :, :page_count].reshape(
            self.layout.batch_size,
            self.layout.num_key_value_heads,
            page_count * self.layout.page_size,
            self.layout.head_dim,
        )[:, :, :seq_len, :]
        return keys, values

    def get_layer_seq_length(self, layer_idx: int) -> int:
        return self.layer_seq_lens[layer_idx]

    def get_seq_length(self) -> int:
        return max(self.layer_seq_lens, default=0)

    def page_table(self, layer_idx: int) -> tuple[int, ...]:
        page_count = self.layout.pages_for_tokens(self.get_layer_seq_length(layer_idx))
        return tuple(
            self.layer_page_table[layer_idx][logical_page_idx]
            for logical_page_idx in range(page_count)
            if self.layer_page_table[layer_idx][logical_page_idx] >= 0
        )

    def summary(self) -> dict[str, int]:
        seq_len = self.get_seq_length()
        return {
            "page_size": self.layout.page_size,
            "num_pages": self.layout.num_pages,
            "used_pages": self.layout.pages_for_tokens(seq_len),
            "allocated_pages": max(self.layer_page_counts, default=0),
            "seq_len": seq_len,
        }
