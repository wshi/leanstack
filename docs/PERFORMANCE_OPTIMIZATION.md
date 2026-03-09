# Performance Optimization

Date verified: 2026-03-07

## Goal

The optimization target is not "make the code cleaner."

The optimization target is:

- maximize `generated_tokens_per_second`
- keep the execution contract fixed to `Qwen/Qwen3-1.7B-Base + GB10 / sm_121`
- keep the active semantics explicitly owned by `leanstack`
- treat `vLLM` as the external throughput baseline

## Current benchmark facts

Primary profile: `decode_64_256`

Long-profile results on the remote GB10 machine:

- earlier semantic runtime: about `29.95 tok/s`
- after KV layout + SDPA + resident measurement: about `36.49 tok/s`
- after request-level RoPE cache + removing per-token host sync: about `37.30 tok/s`
- after switching semantic RMSNorm to the built-in kernel: about `40.81 tok/s`
- after fused `QKV`, fused `gate/up`, sliced RoPE lookup, and compile-friendly decode math: about `44.55 tok/s`
- after decode-only KV append/get fusion, fixed-length request loop tightening, and decode `SDPA -> o_proj` tightening: about `44.61 tok/s`
- warmed `vLLM` baseline on the same profile: about `46.40 tok/s`
- after enforcing an exact `64-token` prompt bucket for the official `decode_64_256` profile: about `44.54 tok/s`
- warmed `vLLM` on the same exact-bucket profile: about `46.06 tok/s`

Short interactive compare (`max_new_tokens=16`) now shows:

- `leanstack`: about `41.27 tok/s`
- `vLLM`: about `37.69 tok/s`

So the repo now has two distinct truths:

- on long steady-state decode, `leanstack` is still behind warmed `vLLM`, but the gap is now only about `3.9%`
- on short interactive requests, the current `leanstack` path can already exceed the `vLLM` side-by-side path

## Optimizations already landed

### 1. KV layout: remove decode-time page concatenation

Files:

- [kv_cache.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/kv_cache.py)

What changed:

- page tensors were rearranged to a view-friendly layout
- `get()` now returns a reshaped live prefix instead of rebuilding K/V with `torch.cat`

Why it matters:

- decode no longer pays repeated small-tensor rebuild cost on every layer and every token

### 2. Attention: switch semantic path to SDPA/GQA

Files:

- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)

What changed:

- semantic attention stopped using explicit `matmul -> softmax -> matmul`
- the active path now goes through `scaled_dot_product_attention`
- grouped-query attention stays explicit

Why it matters:

- it reduces software overhead on the active decode path while preserving semantic ownership

### 3. Resident benchmarking: separate cold materialization from steady-state decode

Files:

- [qwen_explicit_runtime_loop.py](/Users/wei/work/spark/leanstack/experiments/models/qwen_explicit_runtime_loop.py)
- [remote_leanstack_benchmark.sh](/Users/wei/work/spark/leanstack/scripts/remote_leanstack_benchmark.sh)

What changed:

- one runtime materialization now serves multiple requests
- benchmark output reports `warmup_requests` and `resident_requests`
- the measured throughput now reflects steady-state behavior, not repeated cold start

Why it matters:

- without this split, the comparison mostly measured how expensive explicit loading was

### 4. RoPE: request-level position cache

Files:

- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)

What changed:

- `cos/sin` tables are built once per request
- layers consume indexed position embeddings instead of recomputing trigonometric tables every time

Why it matters:

- this removes repeated per-layer, per-step RoPE setup cost from the semantic path

### 5. Benchmark correctness: remove per-token host synchronization

Files:

- [qwen_explicit_runtime_loop.py](/Users/wei/work/spark/leanstack/experiments/models/qwen_explicit_runtime_loop.py)

What changed:

- the decode loop is now timed as a whole segment by default
- optional per-step timings are no longer forced into the main benchmark path

Why it matters:

- the old benchmark path was partially throttling itself by synchronizing the host before and after every token

### 6. RMSNorm: use the built-in kernel

Files:

- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)

What changed:

- semantic RMSNorm now uses `torch.nn.functional.rms_norm`
- the older explicit fallback stays in place only as backup

Why it matters:

- RMSNorm appears multiple times in every layer and decode step
- this change materially improved both long-profile and short-request throughput

### 7. Projection fusion: merge fixed-shape `QKV` and `gate/up`

Files:

- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)

What changed:

- semantic materialization now builds a fused `qkv_proj_weight` for attention
- semantic materialization now builds a fused `gate_up_proj_weight` for the MLP
- the active decode path does one attention input projection instead of three, and one MLP expansion instead of two

Why it matters:

- the model contract is fixed, so these projection groups do not need to stay generic at runtime
- this directly reduces per-layer GEMM launch count on the hottest decode path

### 8. Compile-friendly tensor cores without cudagraph coupling

Files:

- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)
- [qwen_explicit_runtime_loop.py](/Users/wei/work/spark/leanstack/experiments/models/qwen_explicit_runtime_loop.py)

What changed:

- the semantic runtime now routes `QKV` preparation, post-attention MLP work, and greedy head selection through compile-friendly tensor-only helpers
- the compile mode is constrained to fusion-oriented inductor behavior and does not rely on cudagraph replay
- decode now slices RoPE tables by contiguous position ranges instead of rebuilding index tensors

Why it matters:

- this cuts Python dispatch and small-tensor orchestration without breaking the explicit runtime ownership model
- the current long-profile improvement from `40.81` to `44.55 tok/s` comes largely from this step

### 9. Decode-only control-path tightening

Files:

- [kv_cache.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/kv_cache.py)
- [qwen_explicit.py](/Users/wei/work/spark/leanstack/src/leanstack/runtime/qwen_explicit.py)
- [qwen_explicit_runtime_loop.py](/Users/wei/work/spark/leanstack/experiments/models/qwen_explicit_runtime_loop.py)

What changed:

- decode now uses a `KVBlockManager.append_and_get()` path instead of separate `append()` and `get()` calls
- the semantic decode path uses a query-length-1-specific layer helper instead of the more general semantic layer path
- the benchmark/runtime loop now uses a fixed-length fast path when `--ignore-eos` removes stop-token logic
- decode attention now tightens `SDPA -> transpose/reshape -> o_proj` into a smaller helper

Why it matters:

- these changes further reduce control-path overhead on the exact benchmark target
- the measured gain is real but small: `44.55 -> 44.61 tok/s`
- this is a useful signal that the remaining gap is no longer mainly Python/control-path noise

## What still dominates the remaining gap

The active long-profile gap to warmed `vLLM` is now mostly concentrated in these surfaces:

1. logits projection
2. Python-driven token loop structure
3. `down_proj`, `kv_proj`, and other decode linear kernels that still rely on generic eager PyTorch math

The first two are especially important because the current benchmark target is single-request decode. At that shape, control-path overhead and repeated full-vocab projection matter a lot.

Recent measurements also show that incremental Torch-side control-path cleanups now produce only marginal gains. That suggests the next material improvement must come from owning more of the decisive math kernels instead of continuing to shave generic runtime glue.

Recent aggressive experiments also add a second conclusion:

- `static KV` by itself is not enough
- replacing only `lm_head + argmax` is not enough

So the remaining path is not "more micro-cleanup." It is asymmetry-driven specialization.

## Next optimization order

### Priority 1. Logits and sampler path

Current state:

- the semantic path now folds `final_norm -> logits -> argmax` into a tighter helper, but still materializes full-vocab logits each step
- greedy selection is still driven from a generic eager linear over the tied output head

Target:

- reduce or fuse the `final_norm -> logits -> argmax` path
- move this path toward a smaller, more kernel-owned surface

### Priority 2. Decode loop structure

Current state:

- the layer internals are materially more static than before, but the outer decode loop is still Python-driven at the token level

Target:

- move toward a more static decode step contract
- reduce host-side orchestration per emitted token

Concrete direction:

- exact prompt-token buckets for the official profiles
- resident service mode with preallocated buffers and graph-capture-friendly state
- avoid measuring or optimizing against undersized prompt buckets that flatter control-path changes

### Priority 3. Hot-kernel lift into runtime

Current state:

- the repo already has separate BF16 hot-kernel measurements
- runtime still does not directly consume those kernels on the decisive path

Target:

- lift stable wins such as `RMSNorm`, then targeted GEMMs, into the real semantic runtime
- keep the official backend path aligned with `cuTile -> TileIR -> cubin`

Concrete direction:

- offline-pack `QKV`, `gate/up`, `down_proj`, and `lm_head` into the exact layout consumed by the hot runtime
- treat the on-disk checkpoint format as staging only, not the serving format

## Optimization rule

An optimization counts only if it improves one of these:

- long-profile steady-state throughput
- short interactive side-by-side latency/throughput
- both

If a change only makes the codebase smaller or more explicit but does not move benchmark data, it is not a performance optimization.
