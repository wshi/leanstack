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
            "Use PTX only when the public cuTile frontend cannot yet express a required FP4 schedule or instruction pattern; "
            "treat PTX as a temporary escape hatch, not the steady-state interface."
        ),
        sass_stance=(
            "Treat SASS as an inspection and verification artifact. Do not make direct SASS authoring the mainline, "
            "because it is tightly coupled to toolkit and architecture details and lacks a stable public authoring flow."
        ),
        items=(
            GapItem(
                key="fp4-compiler-feasibility",
                title="Prove public FP4 compiler feasibility on sm_121",
                status="in_progress",
                current_state=(
                    "Official external sources show Blackwell FP4 support in the broader NVIDIA stack, and the remote `tileiras` "
                    "binary targets `sm_121`, but the installed public `cuda.tile 1.1.0` frontend only exposes visible dtypes up to FP8."
                ),
                target_state=(
                    "At least one minimal FP4 or NVFP4 GEMM or linear kernel compiles through the public cuTile-native chain "
                    "and runs on the remote GB10."
                ),
                code_surface=(
                    "docs/FP4_COMPILER_GATE.md",
                    "experiments/cutile/vector_add.py",
                    "scripts/remote_verify.sh",
                ),
                next_step=(
                    "Write a minimal FP4 compiler probe, capture TileIR/cubin/SASS artifacts, and record whether the public frontend "
                    "can express the needed dtype and scheduling surface."
                ),
                risk=(
                    "If the public compiler path cannot emit any real FP4 kernel on `sm_121`, the active target is blocked before "
                    "runtime specialization matters."
                ),
            ),
            GapItem(
                key="artifact-contract",
                title="Own the Qwen3-8B-FP4 artifact contract",
                status="missing",
                current_state=(
                    "The repo already handles Qwen config and tokenizer metadata well, but it does not yet own the tensor and scale mapping "
                    "for the `Qwen3-8B-FP4` deployment artifact."
                ),
                target_state=(
                    "The adapter knows how to map FP4 linears, any scale tensors, and higher-precision residual paths without deferring "
                    "that structure to an external runtime."
                ),
                code_surface=(
                    "src/leanstack/model_registry.py",
                    "docs/MODEL_TARGETS.md",
                    "scripts/remote_qwen_fetch.sh",
                ),
                next_step=(
                    "Fetch or relay the target artifact, inspect its metadata layout, and codify the tensor and scale contract in the adapter."
                ),
                risk=(
                    "Without explicit artifact ownership, the runtime cannot know which kernels need FP4 inputs, where scales live, or how "
                    "to keep the execution path static."
                ),
            ),
            GapItem(
                key="semantic-retarget",
                title="Retarget the legacy Qwen semantic path to the 8B FP4 contract",
                status="in_progress",
                current_state=(
                    "The repo already contains a legacy explicit Qwen path for `Qwen3-32B BF16`, including adapter-owned block semantics, "
                    "explicit weight staging, and a leanstack-owned KV path."
                ),
                target_state=(
                    "The same ownership pattern is ported to `Qwen3-8B` semantics with the FP4 artifact layout replacing the older BF16 assumption."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/models/qwen_semantic_block_probe.py",
                    "experiments/models/qwen_explicit_runtime_loop.py",
                ),
                next_step=(
                    "Shrink the geometry to the 8B contract, replace BF16 linear assumptions with FP4-aware layout rules, and keep `transformers` "
                    "only as a correctness oracle."
                ),
                risk=(
                    "If the old semantic path is copied forward without retargeting the artifact contract, the repo will keep optimizing a legacy path "
                    "instead of the active thesis."
                ),
            ),
            GapItem(
                key="residency-and-kv",
                title="Rebuild residency and KV layout for the smaller FP4 target",
                status="missing",
                current_state=(
                    "The repo has a legacy page-based KV manager and residency logic that were shaped around `Qwen3-32B BF16`."
                ),
                target_state=(
                    "A smaller residency plan and KV contract are specialized for `Qwen3-8B-FP4` on GB10, with static layout decisions "
                    "and no hidden placement heuristics."
                ),
                code_surface=(
                    "src/leanstack/runtime/kv_cache.py",
                    "src/leanstack/runtime/qwen_explicit.py",
                    "src/leanstack/runtime/engine.py",
                ),
                next_step=(
                    "Define the 8B residency plan, page geometry, and transfer policy after the FP4 artifact contract is known."
                ),
                risk=(
                    "If residency and KV layout are still inherited from the 32B BF16 reference work, the smaller FP4 target will not realize "
                    "its expected simplicity or throughput advantages."
                ),
            ),
            GapItem(
                key="kernel-catalog",
                title="Build the FP4-first kernel catalog",
                status="missing",
                current_state=(
                    "No FP4 kernel path is proven yet, and the legacy Qwen path still relies on eager PyTorch math for active semantics."
                ),
                target_state=(
                    "A compact kernel catalog exists for FP4 linears or GEMMs, dequant or scale epilogues, RMSNorm, RoPE, GQA prefill, "
                    "GQA decode, gated MLP, logits projection, and sampling."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/cutile/vector_add.py",
                    "docs/FP4_COMPILER_GATE.md",
                ),
                next_step=(
                    "After the compiler probe succeeds, start with the decisive FP4 linear path, then lower norms, RoPE, and the rest of the decode path."
                ),
                risk=(
                    "If the kernel catalog never clears the FP4 linear and decode hot paths, the new target will not produce a meaningful performance case."
                ),
            ),
            GapItem(
                key="benchmark-gate",
                title="Benchmark only after the FP4 runtime exists",
                status="in_progress",
                current_state=(
                    "The benchmark harness exists, but the legacy `Qwen3-32B` path is too slow to produce a meaningful specialized-stack comparison."
                ),
                target_state=(
                    "The first real benchmark table measures the active `Qwen3-8B-FP4 + GB10` contract and records a clear go / no-go conclusion."
                ),
                code_surface=(
                    "src/leanstack/benchmark.py",
                    "scripts/remote_leanstack_benchmark.sh",
                    "scripts/render_benchmark_report.py",
                ),
                next_step=(
                    "Keep benchmark scripts ready, but do not treat any result as dispositive until the FP4 compiler gate and first 8B runtime slice are working."
                ),
                risk=(
                    "Benchmarking too early will compare a legacy reference path against mature frameworks and produce the wrong project conclusion."
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
