from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import get_benchmark_profile, list_benchmark_profiles
from .gap_registry import get_gap_report
from .model_registry import get_model_spec, list_models
from .plan import render_plan
from .remote import default_remote_script, parse_remote_script, run_remote_bash
from .runtime.engine import build_runtime_blueprint, build_static_inference_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leanstack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-plan", help="Print the phased execution plan.")

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
    bench_profile.add_argument("--profile", default="single_stream_short")
    bench_profile.add_argument("--format", choices=("text", "json", "shell"), default="text")
    bench_profile.set_defaults(handler=handle_show_benchmark_profile)

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


def _resolve_endpoint(remote_script: Path):
    if not remote_script.exists():
        raise FileNotFoundError(
            f"remote script not found: {remote_script}. Pass --remote-script or set TILEPILOT_REMOTE_SCRIPT."
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
