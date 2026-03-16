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
            "Use PTX only when diagnosing why the cuTile path misses a decisive kernel shape; "
            "treat PTX as a narrow diagnostic escape hatch, not the official benchmark backend."
        ),
        sass_stance=(
            "Treat SASS as an inspection and verification artifact. Do not make direct SASS authoring the mainline, "
            "because it is tightly coupled to toolkit and architecture details and lacks a stable public authoring flow."
        ),
        items=(
            GapItem(
                key="precision-gate",
                title="Use the executable precision gate as the source of truth",
                status="completed",
                current_state=(
                    "The remote precision gate now returns a real result: BF16 clears on sm_121, FP8 reaches the compiler but fails "
                    "TileIR verification for the current float8 vector-add probe, and FP4 lacks a complete public frontend surface."
                ),
                target_state=(
                    "BF16 remains the active first precision target until the executable precision gate turns positive for a narrower format."
                ),
                code_surface=(
                    "experiments/cutile/precision_gate.py",
                    "experiments/cutile/torch_vector_add.py",
                    "experiments/cutile/fp4_compiler_gate.py",
                    "scripts/remote_precision_gate.sh",
                ),
                next_step=(
                    "Keep the BF16 probe green, and only retry FP8 or FP4 after adding a better narrow-precision kernel probe or a PTX wedge."
                ),
                risk=(
                    "If the repo stops rerunning the precision gate after probe or toolchain changes, it can drift back into unsupported precision assumptions."
                ),
            ),
            GapItem(
                key="checkpoint-contract",
                title="Own the Qwen3-4B-Base BF16 checkpoint contract",
                status="in_progress",
                current_state=(
                    "The repo already handles Qwen config and tokenizer metadata well, and the active target is now the public `Qwen/Qwen3-4B-Base` BF16 checkpoint."
                ),
                target_state=(
                    "The adapter knows the exact tensor naming, checkpoint layout, and residency assumptions for the BF16 Qwen3-4B-Base target without borrowing runtime ownership."
                ),
                code_surface=(
                    "src/leanstack/model_registry.py",
                    "docs/MODEL_TARGETS.md",
                    "scripts/remote_qwen_fetch.sh",
                ),
                next_step=(
                    "Use the existing `Qwen/Qwen3-4B-Base` fetch path as the canonical checkpoint source and codify the BF16 tensor contract in the adapter."
                ),
                risk=(
                    "Without explicit checkpoint ownership, later kernel and residency work will still depend on framework assumptions."
                ),
            ),
            GapItem(
                key="semantic-retarget",
                title="Retarget the legacy Qwen semantic path to the 4B-Base BF16 contract",
                status="in_progress",
                current_state=(
                    "The repo already contains a legacy explicit Qwen path for `Qwen3-32B BF16`, including adapter-owned block semantics, "
                    "explicit weight staging, and a leanstack-owned KV path."
                ),
                target_state=(
                    "The same ownership pattern is ported to `Qwen3-4B-Base` BF16, with clearer geometry and a clean checkpoint contract replacing the older 32B assumptions."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/models/qwen_semantic_block_probe.py",
                    "experiments/models/qwen_explicit_runtime_loop.py",
                ),
                next_step=(
                    "Retarget geometry to the 4B-Base contract, keep BF16 linears explicit, and preserve `transformers` only as a correctness oracle."
                ),
                risk=(
                    "If the old semantic path is copied forward without retargeting the geometry, the repo will keep optimizing the wrong model size."
                ),
            ),
            GapItem(
                key="residency-and-kv",
                title="Rebuild residency and KV layout for the smaller BF16 target",
                status="in_progress",
                current_state=(
                    "The repo now has a real page-table-backed KV manager, but the broader runtime shape is still inherited from the 32B reference work."
                ),
                target_state=(
                    "A dedicated residency plan and KV contract are specialized for `Qwen3-4B-Base` BF16 on GB10, with static layout decisions and no hidden placement heuristics."
                ),
                code_surface=(
                    "src/leanstack/runtime/kv_cache.py",
                    "src/leanstack/runtime/qwen_explicit.py",
                    "src/leanstack/runtime/engine.py",
                ),
                next_step=(
                    "Define the 4B-Base residency plan, page geometry, and transfer policy around the BF16 checkpoint rather than the larger legacy path."
                ),
                risk=(
                    "If residency and KV layout remain shaped by the 32B reference work, the smaller BF16 target will not realize its expected throughput advantage."
                ),
            ),
            GapItem(
                key="kernel-catalog",
                title="Build the BF16-first kernel catalog",
                status="missing",
                current_state=(
                    "BF16 compiles and runs through the public cuTile path for the minimal vector-add probe, but the Qwen runtime still relies on eager PyTorch math."
                ),
                target_state=(
                    "A compact BF16 kernel catalog exists for the 4B-Base hot path: GEMM, RMSNorm, RoPE, GQA prefill, GQA decode, gated MLP, logits projection, and sampling."
                ),
                code_surface=(
                    "src/leanstack/runtime/qwen_explicit.py",
                    "experiments/cutile/torch_vector_add.py",
                    "experiments/cutile/precision_gate.py",
                ),
                next_step=(
                    "Start with BF16 linears, norms, and RoPE on the cuTile path, then revisit FP8 or FP4 only after the BF16 runtime is benchmarkable."
                ),
                risk=(
                    "If the BF16 runtime never leaves eager PyTorch math, the project will still lack a fair specialized-stack performance comparison."
                ),
            ),
            GapItem(
                key="benchmark-gate",
                title="Benchmark only after the 4B-Base BF16 runtime exists",
                status="in_progress",
                current_state=(
                    "The benchmark harness exists, and the precision gate now recommends BF16 as the active target, but the runtime is not yet rebuilt around Qwen3-4B-Base BF16."
                ),
                target_state=(
                    "The first real benchmark table measures the active `Qwen3-4B-Base BF16 + GB10` contract and records a clear go / no-go conclusion."
                ),
                code_surface=(
                    "src/leanstack/benchmark.py",
                    "scripts/remote_leanstack_benchmark.sh",
                    "scripts/render_benchmark_report.py",
                ),
                next_step=(
                    "Rebuild the runtime around Qwen3-4B-Base BF16 first, then run the staged comparison protocol against exact-format BF16 external frameworks."
                ),
                risk=(
                    "Benchmarking too early will compare a partially retargeted runtime against mature frameworks and produce the wrong project conclusion."
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
