from __future__ import annotations

from dataclasses import dataclass

from .model_registry import get_model_spec


@dataclass(frozen=True)
class GapItem:
    key: str
    title: str
    status: str
    current_state: str
    target_state: str
    code_surface: tuple[str, ...]
    next_step: str
    risk: str


@dataclass(frozen=True)
class GapReport:
    model_key: str
    target_gpu: str
    default_codegen_path: str
    ptx_fallback: str
    sass_stance: str
    items: tuple[GapItem, ...]

    def render(self) -> str:
        spec = get_model_spec(self.model_key)
        lines = [f"Implementation gaps for {spec.family} ({spec.key})"]
        lines.append(f"Target GPU: {self.target_gpu}")
        lines.append("Compiler path:")
        lines.append(f"- Preferred: {self.default_codegen_path}")
        lines.append(f"- PTX fallback: {self.ptx_fallback}")
        lines.append(f"- SASS stance: {self.sass_stance}")
        lines.append("Gaps:")
        for idx, item in enumerate(self.items, start=1):
            lines.append(f"{idx}. {item.title} [{item.status}]")
            lines.append(f"   Current: {item.current_state}")
            lines.append(f"   Target: {item.target_state}")
            lines.append(f"   Code surface: {', '.join(item.code_surface)}")
            lines.append(f"   Next: {item.next_step}")
            lines.append(f"   Risk: {item.risk}")
        return "\n".join(lines)


GAP_REGISTRY: dict[str, GapReport] = {
    "qwen": GapReport(
        model_key="qwen",
        target_gpu="NVIDIA GB10 / compute capability sm_121 on the remote DGX Spark machine",
        default_codegen_path=(
            "Keep the main path at `Qwen adapter -> cuTile Python DSL -> TileIR/tilebc -> tileiras -> cubin`, "
            "then inspect PTX and SASS artifacts for the hot kernels."
        ),
        ptx_fallback=(
            "Use PTX only when cuTile/TileIR cannot yet express a required scheduling or instruction pattern; "
            "treat PTX as a temporary escape hatch, not the steady-state interface."
        ),
        sass_stance=(
            "Treat SASS as an inspection and verification artifact. Do not make direct SASS authoring the mainline, "
            "because it is tightly coupled to toolkit and architecture details and lacks a stable public authoring flow."
        ),
        items=(
            GapItem(
                key="semantic-ownership",
                title="Replace borrowed Qwen semantics with adapter-owned operators",
                status="in_progress",
                current_state=(
                    "The explicit path owns weight staging and device placement, but it still calls "
                    "`Qwen3DecoderLayer`, `Qwen3RotaryEmbedding`, `Qwen3RMSNorm`, and `DynamicCache` from `transformers`."
                ),
                target_state=(
                    "The Qwen adapter owns RMSNorm, RoPE, GQA attention, MLP, final norm, output projection, and cache semantics "
                    "without importing model execution behavior from `transformers`."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/models/qwen_explicit_block_probe.py",
                    "experiments/models/qwen_explicit_stack_probe.py",
                ),
                next_step=(
                    "Split `Qwen3DecoderLayer` into explicit adapter stages: input norm, QKV projections, rotary application, "
                    "attention core, output projection, post-attention norm, and gated MLP."
                ),
                risk=(
                    "If the adapter boundary stays inside `transformers`, correctness remains easy but kernel substitution stays blocked."
                ),
            ),
            GapItem(
                key="residency-and-layout",
                title="Turn per-layer staging into a full-model residency plan",
                status="in_progress",
                current_state=(
                    "Weights are staged shard-by-shard from `safetensors` and materialized layer by layer onto GPU, "
                    "which is good enough for probes but not yet a production execution layout."
                ),
                target_state=(
                    "A fixed Qwen3-32B + GB10 layout defines which tensors live permanently on GPU, how KV pages are allocated, "
                    "and how prefill/decode reuse avoids hidden CPU traffic."
                ),
                code_surface=("src/leanstack/runtime/qwen_explicit.py",),
                next_step=(
                    "Add a model-wide residency planner for 64 layers, explicit KV page geometry, and pinned-CPU-to-GPU transfer policy."
                ),
                risk=(
                    "Without a fixed residency plan, later performance work will be polluted by placement and transfer noise."
                ),
            ),
            GapItem(
                key="kv-cache",
                title="Replace `DynamicCache` with a static paged KV manager",
                status="missing",
                current_state=(
                    "Prefill and decode reuse `transformers.cache_utils.DynamicCache`, so cache layout and growth policy are still framework-owned."
                ),
                target_state=(
                    "The runtime owns a paged KV structure specialized for Qwen3-32B GQA geometry on GB10, including block allocation, "
                    "reuse, and deterministic decode addressing."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "src/leanstack/runtime/engine.py",
                ),
                next_step=(
                    "Define a first `KVPageLayout` and `KVBlockManager` API, then swap the stack probe from `DynamicCache` to explicit page buffers."
                ),
                risk=(
                    "Attention kernels cannot become hardware-near while cache layout remains hidden behind a general-purpose framework cache."
                ),
            ),
            GapItem(
                key="kernel-catalog",
                title="Replace torch-backed math with a Qwen kernel catalog",
                status="missing",
                current_state=(
                    "Current explicit probes still inherit dense math from PyTorch-backed `transformers` modules."
                ),
                target_state=(
                    "A minimal kernel catalog exists for RMSNorm, rotary application, QKV/O projections, GQA prefill, GQA decode, "
                    "gated MLP, final norm, logits projection, and greedy sampling."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/cutile/vector_add.py",
                ),
                next_step=(
                    "Start with the non-controversial kernels first: RMSNorm, RoPE, and greedy sampler; then bring up GQA prefill/decode, "
                    "which is the highest-value hotspot."
                ),
                risk=(
                    "The attention and GEMM path may expose performance gaps in the current cuTile/TileIR toolchain on `sm_121`."
                ),
            ),
            GapItem(
                key="compiler-packaging",
                title="Move from probe-time code paths to repeatable `sm_121` compiler artifacts",
                status="missing",
                current_state=(
                    "The repo validates cuTile with a smoke kernel and captures compiler artifacts, but the Qwen execution path is not yet backed "
                    "by an AOT-compiled kernel bundle."
                ),
                target_state=(
                    "Each hot kernel has a repeatable `sm_121` compilation path that emits TileIR, PTX/cubin, and SASS artifacts into the remote workspace."
                ),
                code_surface=(
                    "scripts/remote_verify.sh",
                    "experiments/cutile/vector_add.py",
                ),
                next_step=(
                    "Introduce a kernel bundle layout and compile manifest so each Qwen kernel can be built, hashed, and validated independently."
                ),
                risk=(
                    "Without stable compiler packaging, it is too easy to confuse code changes, toolkit changes, and architecture effects."
                ),
            ),
            GapItem(
                key="scheduler-and-loop",
                title="Upgrade probes into a deterministic runtime loop",
                status="missing",
                current_state=(
                    "The current probes validate one forward/prefill/decode path, but there is no explicit runtime loop for request admission, "
                    "prefill scheduling, decode stepping, or stop handling."
                ),
                target_state=(
                    "A small runtime loop performs deterministic single-request prefill/decode first, then expands to comparable batching rules for benchmark work."
                ),
                code_surface=(
                    "src/leanstack/runtime/engine.py",
                    "experiments/models/qwen_explicit_stack_probe.py",
                ),
                next_step=(
                    "Use the multi-layer probe as the seed of a full 64-layer decode loop, then add stop-token handling and greedy decode accounting."
                ),
                risk=(
                    "Benchmarking against vLLM or SGLang before this loop exists would mostly measure missing runtime surface, not the core compiler/runtime thesis."
                ),
            ),
        ),
    ),
}


def get_gap_report(model_key: str) -> GapReport:
    normalized = model_key.strip().lower()
    try:
        return GAP_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(GAP_REGISTRY))
        raise KeyError(f"no structured gap report for '{model_key}'. Supported models: {supported}") from exc
