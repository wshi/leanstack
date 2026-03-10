from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .appliance import render_appliance_reset, render_leanpack_plan, render_leanserve_plan
from .benchmark import get_benchmark_profile, list_benchmark_profiles
from .comparison import render_comparison_plan
from .gap_registry import get_gap_report
from .hot_kernels import get_hot_kernel_case, list_hot_kernel_cases
from .model_registry import get_model_spec, list_models
from .plan import render_plan
from .remote import default_remote_script, parse_remote_script, run_remote_bash
from .runtime.engine import build_runtime_blueprint, build_static_inference_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leanstack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-plan", help="Print the phased execution plan.")
    subparsers.add_parser("show-comparison-plan", help="Print the staged comparison protocol.")

    appliance = subparsers.add_parser("show-appliance-reset", help="Print the first-principles appliance reset.")
    appliance.add_argument("--model", default="qwen")
    appliance.set_defaults(handler=handle_show_appliance_reset)

    leanpack = subparsers.add_parser("show-leanpack-plan", help="Print the offline serving-artifact plan.")
    leanpack.add_argument("--model", default="qwen")
    leanpack.set_defaults(handler=handle_show_leanpack_plan)

    leanserve = subparsers.add_parser("show-leanserve-plan", help="Print the resident decode appliance plan.")
    leanserve.add_argument("--model", default="qwen")
    leanserve.set_defaults(handler=handle_show_leanserve_plan)

    build_leanpack = subparsers.add_parser("build-leanpack", help="Build a serving-only packed artifact.")
    build_leanpack.add_argument("--model", default="qwen")
    build_leanpack.add_argument("--model-path", required=True)
    build_leanpack.add_argument("--output-dir", type=Path, required=True)
    build_leanpack.add_argument("--manifest-only", action="store_true")
    build_leanpack.add_argument("--overwrite", action="store_true")
    build_leanpack.set_defaults(handler=handle_build_leanpack)

    fit_draft_head = subparsers.add_parser(
        "fit-draft-head",
        help="Fit and store an auxiliary draft head for speculative decode.",
    )
    fit_draft_head.add_argument("--model-path", required=True)
    fit_draft_head.add_argument("--pack-dir", type=Path, required=True)
    fit_draft_head.add_argument("--draft-layer-count", type=int, required=True)
    fit_draft_head.add_argument("--key", required=True)
    fit_draft_head.add_argument("--chunk-tokens", type=int, default=128)
    fit_draft_head.add_argument("--max-chunks", type=int, default=32)
    fit_draft_head.add_argument("--ridge-lambda", type=float, default=0.1)
    fit_draft_head.add_argument("--calibration-mode", choices=("prefill", "decode"), default="prefill")
    fit_draft_head.add_argument("--decode-steps", type=int, default=16)
    fit_draft_head.add_argument("--device", default="cuda:0")
    fit_draft_head.add_argument("--dtype", default="bfloat16")
    fit_draft_head.add_argument("--repo-root", type=Path, default=Path.cwd())
    fit_draft_head.set_defaults(handler=handle_fit_draft_head)

    inspect_leanpack = subparsers.add_parser("inspect-leanpack", help="Inspect a packed serving artifact.")
    inspect_leanpack.add_argument("--pack-dir", type=Path, required=True)
    inspect_leanpack.set_defaults(handler=handle_inspect_leanpack)

    leanserve_layout = subparsers.add_parser(
        "show-leanserve-layout",
        help="Print the resident appliance layout for a packed artifact.",
    )
    leanserve_layout.add_argument("--model", default="qwen")
    leanserve_layout.add_argument("--pack-dir", type=Path, required=True)
    leanserve_layout.add_argument("--device", default="cuda:0")
    leanserve_layout.add_argument("--dtype", default="")
    leanserve_layout.add_argument("--page-size", type=int, default=16)
    leanserve_layout.add_argument("--batch-size", type=int, default=1)
    leanserve_layout.set_defaults(handler=handle_show_leanserve_layout)

    models = subparsers.add_parser("list-models", help="List supported adapter targets.")
    models.set_defaults(handler=handle_list_models)

    blueprint = subparsers.add_parser("show-blueprint", help="Print the runtime blueprint for a model.")
    blueprint.add_argument("--model", default="qwen")
    blueprint.set_defaults(handler=handle_show_blueprint)

    contract = subparsers.add_parser("show-contract", help="Print the static inference contract for a model.")
    contract.add_argument("--model", default="qwen")
    contract.set_defaults(handler=handle_show_contract)

    gaps = subparsers.add_parser("show-gaps", help="Print the implementation gaps for a model.")
    gaps.add_argument("--model", default="qwen")
    gaps.set_defaults(handler=handle_show_gaps)

    bench_profiles = subparsers.add_parser("list-benchmark-profiles", help="List supported benchmark profiles.")
    bench_profiles.set_defaults(handler=handle_list_benchmark_profiles)

    bench_profile = subparsers.add_parser("show-benchmark-profile", help="Print one benchmark profile.")
    bench_profile.add_argument("--profile", default="decode_64_256")
    bench_profile.add_argument("--format", choices=("text", "json", "shell"), default="text")
    bench_profile.set_defaults(handler=handle_show_benchmark_profile)

    hot_cases = subparsers.add_parser("list-hot-kernel-cases", help="List supported cuTile hot-kernel cases.")
    hot_cases.add_argument("--default-only", action="store_true")
    hot_cases.set_defaults(handler=handle_list_hot_kernel_cases)

    hot_case = subparsers.add_parser("show-hot-kernel-case", help="Print one cuTile hot-kernel case.")
    hot_case.add_argument("--case", default="q_proj_prefill64")
    hot_case.add_argument("--format", choices=("text", "json", "shell"), default="text")
    hot_case.set_defaults(handler=handle_show_hot_kernel_case)

    remote_env = subparsers.add_parser("remote-env", help="Probe the configured remote environment.")
    remote_env.add_argument("--remote-script", type=Path, default=default_remote_script())
    remote_env.set_defaults(handler=handle_remote_env)

    bootstrap = subparsers.add_parser("bootstrap-remote", help="Create the remote workspace layout.")
    bootstrap.add_argument("--remote-script", type=Path, default=default_remote_script())
    bootstrap.set_defaults(handler=handle_bootstrap_remote)

    return parser


def handle_list_models(_: argparse.Namespace) -> int:
    for spec in list_models():
        print(f"{spec.key}: {spec.family}")
        print(f"  loader hint: {spec.loader_hint}")
    return 0


def handle_show_appliance_reset(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(render_appliance_reset(spec))
    return 0


def handle_show_leanpack_plan(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(render_leanpack_plan(spec))
    return 0


def handle_show_leanserve_plan(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(render_leanserve_plan(spec))
    return 0


def handle_build_leanpack(args: argparse.Namespace) -> int:
    from .pack import build_qwen_leanpack

    spec = get_model_spec(args.model)
    if spec.key != "qwen":
        raise ValueError(f"leanpack builder not implemented for model={spec.key}")
    manifest = build_qwen_leanpack(
        model=spec,
        model_path=args.model_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        write_tensors=not args.manifest_only,
    )
    print(json.dumps(manifest.as_payload(), indent=2))
    return 0


def handle_fit_draft_head(args: argparse.Namespace) -> int:
    from .draft_head import fit_qwen_draft_projection

    result = fit_qwen_draft_projection(
        model_path=args.model_path,
        pack_dir=args.pack_dir,
        draft_layer_count=args.draft_layer_count,
        key=args.key,
        chunk_tokens=args.chunk_tokens,
        max_chunks=args.max_chunks,
        ridge_lambda=args.ridge_lambda,
        calibration_mode=args.calibration_mode,
        decode_steps=args.decode_steps,
        device=args.device,
        dtype=args.dtype,
        repo_root=args.repo_root,
    )
    print(json.dumps(result.as_payload(), indent=2))
    return 0


def handle_inspect_leanpack(args: argparse.Namespace) -> int:
    from .leanserve import load_leanpack_artifact

    artifact = load_leanpack_artifact(args.pack_dir)
    print(artifact.describe())
    return 0


def handle_show_leanserve_layout(args: argparse.Namespace) -> int:
    from .leanserve import build_leanserve_appliance

    spec = get_model_spec(args.model)
    appliance = build_leanserve_appliance(
        model=spec,
        pack_dir=args.pack_dir,
        device=args.device,
        dtype=args.dtype or None,
        page_size=args.page_size,
        batch_size=args.batch_size,
    )
    print(appliance.render())
    return 0


def handle_show_blueprint(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(build_runtime_blueprint(spec).render())
    return 0


def handle_show_contract(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(build_static_inference_contract(spec).render())
    return 0


def handle_show_gaps(args: argparse.Namespace) -> int:
    print(get_gap_report(args.model).render())
    return 0


def handle_list_benchmark_profiles(_: argparse.Namespace) -> int:
    for profile in list_benchmark_profiles():
        print(profile.render())
    return 0


def handle_show_benchmark_profile(args: argparse.Namespace) -> int:
    profile = get_benchmark_profile(args.profile)
    if args.format == "json":
        print(json.dumps(profile.as_payload(), indent=2))
    elif args.format == "shell":
        print(profile.render_shell())
    else:
        print(profile.render())
    return 0


def handle_list_hot_kernel_cases(args: argparse.Namespace) -> int:
    for case in list_hot_kernel_cases(default_only=args.default_only):
        print(case.render())
    return 0


def handle_show_hot_kernel_case(args: argparse.Namespace) -> int:
    case = get_hot_kernel_case(args.case)
    if args.format == "json":
        print(case.render_json())
    elif args.format == "shell":
        print(case.render_shell())
    else:
        print(case.render())
    return 0


def _resolve_endpoint(remote_script: Path):
    if not remote_script.exists():
        raise FileNotFoundError(
            f"remote script not found: {remote_script}. Pass --remote-script or set LEANSTACK_REMOTE_SCRIPT."
        )
    return parse_remote_script(remote_script)


def handle_remote_env(args: argparse.Namespace) -> int:
    endpoint = _resolve_endpoint(args.remote_script)
    result = run_remote_bash(
        endpoint,
        """
set -euo pipefail
echo "host=$(hostname)"
echo "cwd=$(pwd)"
if [ -d /home/pto/venv-cutile ]; then
  . /home/pto/venv-cutile/bin/activate
fi
python3 - <<'PY'
import importlib
import platform

print(f"python={platform.python_version()}")
for module in ("cuda.tile", "cupy", "torch", "transformers"):
    try:
        pkg = importlib.import_module(module)
        print(f"{module}={getattr(pkg, '__version__', 'present')}")
    except Exception as exc:
        print(f"{module}=MISSING ({type(exc).__name__}: {exc})")
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
""".strip(),
    )
    sys.stdout.write(result.stdout)
    return 0


def handle_bootstrap_remote(args: argparse.Namespace) -> int:
    endpoint = _resolve_endpoint(args.remote_script)
    result = run_remote_bash(
        endpoint,
        f"""
set -euo pipefail
REMOTE_HOME={endpoint.workspace}
mkdir -p "$REMOTE_HOME"/{{repo,artifacts,logs,models,tmp}}
find "$REMOTE_HOME" -maxdepth 1 -mindepth 1 -type d | sort
""".strip(),
    )
    sys.stdout.write(result.stdout)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "handler"):
        return args.handler(args)

    if args.command == "show-plan":
        print(render_plan())
        return 0

    if args.command == "show-comparison-plan":
        print(render_comparison_plan())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
