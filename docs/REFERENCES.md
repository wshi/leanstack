# References

Date verified: 2026-03-07

## Agentic CUDA work

- [CUDA Agent project page](https://cuda-agent.github.io/)
  - Primary source for the ByteDance Seed + Tsinghua AIR work on agentic CUDA kernel generation.
  - Relevant here because it frames CUDA optimization as an agent environment with compile, verify, and profile loops.

- [CUDA Agent GitHub repository](https://github.com/BytedTsinghua-SIA/CUDA-Agent)
  - Primary source for the released `agent_workdir`, `SKILL.md`, verification flow, and profiling loop against `torch.compile`.
  - Relevant here because `leanstack` should adopt the same standard: visible constraints, visible tools, visible performance checks.

- [Claude Code Ports NVIDIA CUDA to AMD ROCm in 30 Minutes](https://techstrong.ai/features/claude-code-ports-nvidia-cuda-to-amd-rocm-in-30-minutes/)
  - Secondary source.
  - Useful as an industry signal that agentic systems can compress GPU-software migration work.
  - Not benchmark evidence by itself; the article explicitly notes that the underlying codebase complexity was not disclosed.

## NVIDIA compiler stack and low-level references

- [NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
  - Primary source for the public cuTile Python surface.
  - Relevant because `leanstack` is trying to keep the primary authoring path in the public `cuTile` stack instead of importing a large runtime.

- [NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
  - Primary source for TileIR tooling, including `tilebc` and `tileiras`.
  - Relevant because the repo depends on a visible `TileIR -> cubin` lowering chain.

- [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html)
  - Official NVIDIA reference for current Blackwell-family support and the CuTe DSL positioning.
  - Useful here because it exposes the current NVIDIA view of Python-DSL-to-kernel authoring and explicitly lists Blackwell-family targets, including DGX Spark / `12.1`.

- [Blackwell functionality in CUTLASS](https://docs.nvidia.com/cutlass/latest/overview.html#blackwell-functionality)
  - Official NVIDIA reference for Blackwell narrow-precision support in the broader CUTLASS stack.
  - Relevant because it is part of the external evidence that FP8 or FP4 support exists somewhere in the broader NVIDIA stack even though the public `cuda.tile` path still has active gaps.

- [CuTe DSL framework integration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html)
  - Official reference for how the NVIDIA Python DSL interoperates with framework tensors and exported kernels.
  - Relevant because `leanstack` needs to move from framework-owned execution to framework-assisted tensor handoff.

- [CuTe DSL AOT compilation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/aot_compilation.html)
  - Official reference for ahead-of-time kernel export.
  - Relevant because stable `sm_121` kernel packaging is necessary before benchmark claims are trustworthy.

- [Blackwell Compatibility Guide for CUDA Applications](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
  - Official reference for PTX/cubin compatibility behavior and architecture-accelerated feature constraints.
  - Relevant because it frames when PTX is sufficient and when architecture-specific compilation matters.

- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
  - Official reference for cubin/PTX inspection and `nvdisasm`.
  - Relevant because `leanstack` treats SASS as a verification artifact even when it is not the primary authoring layer.

## Framework baselines and design references

- [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
  - Official design reference for vLLM's multi-process architecture.

- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
  - Official reference for scheduler behavior, chunked prefill, preemption, and CPU resource requirements.

- [vLLM Compressed Tensors](https://docs.vllm.ai/en/stable/features/quantization/compressed_tensors.html)
  - Official vLLM reference for compressed-tensor and quantized deployment paths.
  - Relevant because exact-format or near-format narrow-precision comparisons must be labeled precisely.

- [vLLM Paged Attention](https://docs.vllm.ai/en/stable/design/paged_attention.html)
  - Official but historical design note.
  - Useful as a conceptual kernel reference only; the page states it no longer describes the current code exactly.

- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
  - Official reference for a compact C/C++ inference system and low-dependency deployment philosophy.

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
  - Official reference for how a relatively small runtime exposes OpenAI-compatible and cache-aware server endpoints.

- [SGLang documentation](https://docs.sglang.io/)
  - Official reference for SGLang's runtime scope, including RadixAttention, continuous batching, disaggregation, and quantization features.

- [SGLang HiCache](https://docs.sglang.io/advanced_features/hicache.html)
  - Official reference for hierarchical KV caching and long-context serving design.

- [SGLang Quantization](https://docs.sglang.io/advanced_features/quantization.html)
  - Official reference for SGLang quantization paths.
  - Relevant because any future FP8 or FP4 benchmark claims should distinguish exact-format support from other quantized variants.

## Target model and target hardware

- [Qwen/Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
  - Primary public reference for the active semantic and BF16 checkpoint target, including the small dense geometry used for the throughput-first pivot.

- [nvidia/Qwen3-8B-FP4 model card](https://huggingface.co/nvidia/Qwen3-8B-FP4)
  - Secondary reference for the deferred FP4 route and for the negative FP4 gate discussion.

- [Qwen/Qwen3-1.7B-Base on ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B-Base)
  - Primary semantic-base acquisition reference when the remote machine can reach ModelScope more reliably than Hugging Face.

- [NVIDIA Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
  - Primary reference for Blackwell hardware characteristics and Transformer Engine direction.

- [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
  - Primary reference for the GB10 / DGX Spark hardware envelope that shapes the first deployment target.
