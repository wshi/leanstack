# References

Date verified: 2026-03-06

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

## Framework baselines and design references

- [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
  - Official design reference for vLLM's multi-process architecture.

- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
  - Official reference for scheduler behavior, chunked prefill, preemption, and CPU resource requirements.

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

## Target model and target hardware

- [Qwen/Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B)
  - Primary public reference for Qwen3-32B model features, context claims, and thinking / non-thinking usage guidance.

- [Qwen/Qwen3-32B on ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-32B)
  - Primary deployment reference because the remote machine can reach ModelScope even when direct Hugging Face artifact downloads time out.

- [NVIDIA Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
  - Primary reference for Blackwell hardware characteristics and Transformer Engine direction.

- [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
  - Primary reference for the GB10 / DGX Spark hardware envelope that shapes the first deployment target.
