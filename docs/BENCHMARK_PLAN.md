# Benchmark Plan

Date: 2026-03-06

## Question

Can a `cuTile`-native runtime tuned for `Qwen/Qwen3-32B` on Blackwell stay much smaller than current inference frameworks while delivering competitive throughput and latency?

The benchmark is not only about runtime speed.

It is also about whether agent-generated, model-chip-specific software can replace a significant amount of compatibility-oriented stack complexity.

## Primary systems under test

- `leanstack`
- `vLLM`
- `SGLang`

## Secondary reference

- `llama.cpp`

`llama.cpp` is important as a compact-systems reference, but it is not automatically an apples-to-apples throughput baseline because `Qwen3-32B` is commonly deployed there through `GGUF` or quantized variants rather than the same BF16 checkpoint used by `vLLM`, `SGLang`, and `leanstack`.

## Benchmark contract

### Hardware

- same remote machine under `/home/pto/lean`
- same Blackwell-class GB10 environment
- same driver and CUDA environment for all framework runs

### Model

- primary checkpoint: `Qwen/Qwen3-32B`
- primary precision path: BF16
- exact model snapshot and framework versions must be recorded in every report

### Prompting policy

The primary throughput profile should use Qwen3 in non-thinking mode:

- `enable_thinking=False`
- deterministic decode for repeatability unless a benchmark explicitly studies sampling

Reason:

- reasoning mode adds highly variable hidden-thought token counts
- that variance can hide stack effects behind model-behavior effects

## Primary metrics

- generated tokens per second
- time to first token
- end-to-end latency
- peak GPU memory
- steady-state GPU utilization
- CPU process count and thread pressure
- deployment surface area: process shape, launch/config complexity, and runtime dependencies

## Complexity and cost proxies

The research claim also needs non-throughput metrics:

- dependency count in the serving path
- number of long-running processes required
- amount of configuration needed to launch a comparable run
- code surface that must be touched to specialize the stack for Qwen3-32B on Blackwell
- agent token budget spent to reach or revise a runnable path, when that information is available from the working session

These are proxies, not perfect cost measures, but they are necessary if the thesis is that compatibility-heavy stacks carry avoidable overhead.

## First benchmark profiles

### Profile A: Single-stream decode

- one request
- short prompt
- fixed output length
- goal: isolate decode throughput and scheduler overhead

### Profile B: Long-prefill latency

- one request
- long prompt
- shorter output length
- goal: expose prefill cost and KV/cache setup behavior

### Profile C: Small concurrent batch

- several simultaneous requests on one machine
- moderate prompt length
- fixed output length
- goal: compare batching policy and cache management

## Comparison rules

### For `vLLM`

- use official stable docs and record process shape
- treat it as a high-performance framework baseline, not a source of core runtime code
- note where generality or multi-role architecture appears to add operational cost for this narrower target

### For `SGLang`

- use official stable docs and record enabled features explicitly
- disable unrelated features unless they are part of the tested profile
- note which features are irrelevant to the narrow Qwen3-32B + Blackwell target but still shape operational complexity

### For `llama.cpp`

- separate deployment-complexity comparisons from strict BF16 throughput comparisons when formats differ
- use it partly as a compact-systems reference for what a smaller runtime can look like

## Desired output

Every benchmark report should end with:

1. a normalized table for `leanstack`, `vLLM`, and `SGLang`
2. a short note on whether `llama.cpp` was comparable in that run
3. a short section on compatibility tax versus specialization benefit
4. a conclusion on where `leanstack` won, lost, or remained incomplete
5. an explanation of which missing kernels, runtime policies, or deferred compatibility features matter most
