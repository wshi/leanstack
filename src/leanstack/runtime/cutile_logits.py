from __future__ import annotations

from typing import Any

import cuda.tile as ct
import torch


_LOGITS_ARGMAX_KERNELS: dict[tuple[int, int], Any] = {}


def _get_logits_argmax_kernel(vocab_tile: int, hidden_tile: int):
    key = (vocab_tile, hidden_tile)
    if key in _LOGITS_ARGMAX_KERNELS:
        return _LOGITS_ARGMAX_KERNELS[key]

    @ct.kernel
    def _kernel(weight, hidden, tile_scores, tile_argmax, k_tiles: ct.Constant[int]):
        pid_vocab = ct.bid(0)
        acc = ct.zeros((vocab_tile,), dtype=ct.float32)
        for k_block in range(k_tiles):
            weight_tile = ct.load(weight, index=(pid_vocab, k_block), shape=(vocab_tile, hidden_tile))
            hidden_tile_vec = ct.reshape(
                ct.load(hidden, index=(0, k_block), shape=(1, hidden_tile)),
                (hidden_tile,),
            )
            acc = acc + ct.sum(ct.astype(weight_tile, ct.float32) * hidden_tile_vec, axis=1)
        tile_score = ct.reshape(ct.max(acc, axis=0), (1,))
        tile_index = ct.astype(ct.reshape(ct.argmax(acc, axis=0), (1,)), ct.float32)
        ct.store(tile_scores, index=(pid_vocab,), tile=tile_score)
        ct.store(tile_argmax, index=(pid_vocab,), tile=tile_index)

    _LOGITS_ARGMAX_KERNELS[key] = _kernel
    return _kernel


def cutile_logits_argmax(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_tile: int = 128,
    hidden_tile: int = 64,
) -> torch.LongTensor:
    if hidden_states.ndim != 1:
        raise ValueError(f"expected a 1D hidden-state vector, got shape={tuple(hidden_states.shape)}")
    if hidden_states.device.type != "cuda" or lm_head_weight.device.type != "cuda":
        raise ValueError("cuTile logits argmax requires CUDA tensors")

    vocab_size, hidden_size = lm_head_weight.shape
    if vocab_size % vocab_tile != 0:
        raise ValueError(f"vocab_size={vocab_size} must be divisible by vocab_tile={vocab_tile}")
    if hidden_size % hidden_tile != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by hidden_tile={hidden_tile}")

    hidden_matrix = hidden_states.contiguous().view(1, hidden_size)
    tile_count = vocab_size // vocab_tile
    tile_scores = torch.empty((tile_count,), device=hidden_states.device, dtype=torch.float32)
    tile_argmax = torch.empty((tile_count,), device=hidden_states.device, dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    kernel = _get_logits_argmax_kernel(vocab_tile, hidden_tile)
    ct.launch(
        stream,
        (tile_count, 1, 1),
        kernel,
        (lm_head_weight, hidden_matrix, tile_scores, tile_argmax, hidden_size // hidden_tile),
    )
    best_tile = torch.argmax(tile_scores)
    local_index = tile_argmax[best_tile].to(torch.long)
    return (best_tile.to(torch.long) * vocab_tile + local_index).view(1, 1)
