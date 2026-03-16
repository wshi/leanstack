from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_HOST = "pto@kb119856792y.vicp.fun"
REMOTE_PORT = "33402"
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Base"
DEFAULT_MODEL_PATH_FILE = "/home/pto/lean/models/Qwen__Qwen3-4B-Base.path"
DEFAULT_PACK_DIR = "/home/pto/lean/packed/Qwen__Qwen3-4B-Base"
DEFAULT_PROFILE = "decode_64_256"
DEFAULT_RUNTIME_MODE = "semantic"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE = "cuda:0"
OFFICIAL_CONTRACT_ID = "qwen3-4b-base-bf16-gb10-sm121-decode_64_256"
OFFICIAL_HARDWARE = "GB10 / sm_121"
DEFAULT_MODEL_NAME = "qwen3-4b-base"
DEFAULT_VLLM_BASELINE_MODE = "best"
DEFAULT_VLLM_BASELINE_RUNS = 3
DEFAULT_VLLM_VENV = "/home/pto/lean/venv-vllm-cu128"
DEFAULT_PYTHON_DEV_ROOT = "/home/pto/lean/tmp/pydev_probe/extracted"
DEFAULT_VLLM_EXEC = f"{DEFAULT_VLLM_VENV}/bin/vllm"
DEFAULT_VLLM_PYTHON_LAUNCH = f"{DEFAULT_VLLM_VENV}/bin/python3 {DEFAULT_VLLM_EXEC}"
DEFAULT_DECODE_POLICY = "greedy"
DEFAULT_SAMPLING_TEMPERATURE = 0.0
DEFAULT_IGNORE_EOS = True


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


def _run_command(command: list[str], *, env: dict[str, str] | None = None) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        command=tuple(command),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _raise_on_failure(result: CommandResult, label: str) -> None:
    if result.returncode == 0:
        return
    message = [
        f"{label} failed with exit code {result.returncode}",
        f"command: {' '.join(result.command)}",
    ]
    if result.stdout.strip():
        message.append(f"stdout tail:\n{_tail_lines(result.stdout)}")
    if result.stderr.strip():
        message.append(f"stderr tail:\n{_tail_lines(result.stderr)}")
    raise RuntimeError("\n".join(message))


def _tail_lines(text: str, limit: int = 80) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-limit:])


def _extract_last_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if text[idx + end :].strip():
            continue
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object at the end of command output")
        return payload
    raise ValueError("could not find a terminal JSON object in command output")


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return None


def validate_comparison_fairness(
    *,
    vllm: dict[str, Any],
    leanstack: dict[str, Any],
    expected_max_new_tokens: int | None = None,
) -> dict[str, Any]:
    vllm_prompt_tokens = _parse_int(vllm.get("prompt_tokens"))
    leanstack_prompt_tokens = _parse_int(leanstack.get("prompt_tokens"))
    vllm_generated_tokens = _parse_int(vllm.get("generated_tokens"))
    leanstack_generated_tokens = _parse_int(leanstack.get("emitted_tokens"))
    vllm_temperature = _parse_float(vllm.get("temperature"))
    vllm_ignore_eos = _parse_bool(vllm.get("ignore_eos"))
    leanstack_ignore_eos = _parse_bool(leanstack.get("ignore_eos"))
    leanstack_decode_policy = str(leanstack.get("decode_policy") or "").strip().lower()

    target_max_new_tokens = expected_max_new_tokens
    if target_max_new_tokens is None:
        target_max_new_tokens = _parse_int(leanstack.get("max_new_tokens"))

    violations: list[str] = []
    if vllm_temperature is None:
        violations.append("vLLM result is missing `temperature` metadata")
    elif abs(vllm_temperature - DEFAULT_SAMPLING_TEMPERATURE) > 1e-12:
        violations.append(
            f"vLLM temperature must be {DEFAULT_SAMPLING_TEMPERATURE}, got {vllm_temperature}"
        )

    if leanstack_decode_policy != DEFAULT_DECODE_POLICY:
        violations.append(
            f"leanstack decode policy must be `{DEFAULT_DECODE_POLICY}`, got `{leanstack_decode_policy or 'missing'}`"
        )

    if vllm_ignore_eos is None:
        violations.append("vLLM result is missing `ignore_eos` metadata")
    elif vllm_ignore_eos != DEFAULT_IGNORE_EOS:
        violations.append(f"vLLM ignore_eos must be {DEFAULT_IGNORE_EOS}, got {vllm_ignore_eos}")

    if leanstack_ignore_eos is None:
        violations.append("leanstack result is missing `ignore_eos` metadata")
    elif leanstack_ignore_eos != DEFAULT_IGNORE_EOS:
        violations.append(f"leanstack ignore_eos must be {DEFAULT_IGNORE_EOS}, got {leanstack_ignore_eos}")

    if vllm_prompt_tokens is None or leanstack_prompt_tokens is None:
        violations.append("prompt token counts must exist on both sides")
    elif vllm_prompt_tokens != leanstack_prompt_tokens:
        violations.append(
            f"prompt token mismatch: vLLM={vllm_prompt_tokens}, leanstack={leanstack_prompt_tokens}"
        )

    if vllm_generated_tokens is None or leanstack_generated_tokens is None:
        violations.append("generated token counts must exist on both sides")
    elif vllm_generated_tokens != leanstack_generated_tokens:
        violations.append(
            f"generated token mismatch: vLLM={vllm_generated_tokens}, leanstack={leanstack_generated_tokens}"
        )

    if target_max_new_tokens is not None:
        if vllm_generated_tokens is None or vllm_generated_tokens != target_max_new_tokens:
            violations.append(
                f"vLLM generated_tokens must equal max_new_tokens={target_max_new_tokens}, got {vllm_generated_tokens}"
            )
        if leanstack_generated_tokens is None or leanstack_generated_tokens != target_max_new_tokens:
            violations.append(
                f"leanstack emitted_tokens must equal max_new_tokens={target_max_new_tokens}, got {leanstack_generated_tokens}"
            )

    if violations:
        raise RuntimeError(
            "comparison fairness gate failed:\n- " + "\n- ".join(violations)
        )

    return {
        "passed": True,
        "decode_policy": DEFAULT_DECODE_POLICY,
        "temperature": DEFAULT_SAMPLING_TEMPERATURE,
        "ignore_eos": DEFAULT_IGNORE_EOS,
        "prompt_tokens": vllm_prompt_tokens,
        "generated_tokens": vllm_generated_tokens,
        "max_new_tokens": target_max_new_tokens,
    }


def _run_shell_script(script_name: str, env: dict[str, str] | None = None) -> CommandResult:
    script_path = REPO_ROOT / "scripts" / script_name
    return _run_command(["/bin/zsh", str(script_path)], env=env)


def _run_remote_bash(script: str) -> CommandResult:
    remote_command = f"bash -lc {shlex.quote(script)}"
    shell_command = f"ssh {shlex.quote(REMOTE_HOST)} -p {shlex.quote(REMOTE_PORT)} {shlex.quote(remote_command)}"
    return _run_command(["/bin/sh", "-lc", shell_command])


def check_remote_status() -> dict[str, Any]:
    remote_script = r"""
set -euo pipefail
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

path_file = Path("/home/pto/lean/models/Qwen__Qwen3-4B-Base.path")
pack_dir = Path("/home/pto/lean/packed/Qwen__Qwen3-4B-Base")
manifest_file = pack_dir / "manifest.json"
model_ref = path_file.read_text().strip() if path_file.exists() else ""
model_dir = Path(model_ref) if model_ref else Path("")
weight_file = model_dir / "model.safetensors" if model_ref else Path("")
weight_index_file = model_dir / "model.safetensors.index.json" if model_ref else Path("")
fetch = subprocess.run(["pgrep", "-xaf", r"python3 .*fetch_modelscope_snapshot.py"], capture_output=True, text=True)
vllm_ready = subprocess.run(
    ["bash", "-lc", "curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1"],
    capture_output=True,
)

size_bytes = 0
if model_ref and model_dir.exists():
    for path in model_dir.rglob("*"):
        if path.is_file():
            size_bytes += path.stat().st_size

expected_weight_files = []
missing_weight_files = []
weight_shard_sizes = {}
if model_ref and weight_index_file.exists():
    index_payload = json.loads(weight_index_file.read_text())
    expected_weight_files = sorted(set(index_payload.get("weight_map", {}).values()))
    for filename in expected_weight_files:
        shard_path = model_dir / filename
        if shard_path.exists():
            weight_shard_sizes[filename] = shard_path.stat().st_size
        else:
            missing_weight_files.append(filename)

weight_single_file_complete = bool(model_ref and weight_file.exists() and weight_file.stat().st_size > 0)
weight_shards_complete = bool(expected_weight_files) and not missing_weight_files

payload = {
    "path_file_exists": path_file.exists(),
    "path_file": model_ref or None,
    "model_dir_exists": bool(model_ref and model_dir.exists()),
    "model_files": sorted(path.name for path in model_dir.glob("*") if path.is_file()) if model_ref and model_dir.exists() else [],
    "model_size_bytes": size_bytes,
    "weight_file_exists": bool(model_ref and weight_file.exists()),
    "weight_size_bytes": weight_file.stat().st_size if model_ref and weight_file.exists() else 0,
    "weight_index_exists": bool(model_ref and weight_index_file.exists()),
    "expected_weight_files": expected_weight_files,
    "missing_weight_files": missing_weight_files,
    "weight_shard_sizes": weight_shard_sizes,
    "weight_single_file_complete": weight_single_file_complete,
    "weight_shards_complete": weight_shards_complete,
    "pack_dir": str(pack_dir),
    "pack_dir_exists": pack_dir.exists(),
    "pack_manifest_exists": manifest_file.exists(),
    "fetch_processes": [line for line in fetch.stdout.splitlines() if line.strip()],
    "vllm_ready": vllm_ready.returncode == 0,
}
payload["download_complete"] = weight_single_file_complete or weight_shards_complete
payload["pack_ready"] = payload["pack_manifest_exists"]
print(json.dumps(payload))
PY
""".strip()
    result = _run_remote_bash(remote_script)
    _raise_on_failure(result, "remote status probe")
    return json.loads(result.stdout)


def ensure_vllm_ready() -> dict[str, Any]:
    result = _run_shell_script(
        "remote_vllm_serve.sh",
        env={
            "VLLM_VENV": DEFAULT_VLLM_VENV,
            "MODEL_ID": DEFAULT_MODEL_ID,
            "SERVED_MODEL_NAME": DEFAULT_MODEL_NAME,
            "PYTHON_DEV_ROOT": DEFAULT_PYTHON_DEV_ROOT,
        },
    )
    _raise_on_failure(result, "vLLM serve")
    return {
        "status": "ready",
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr) if result.stderr.strip() else "",
    }


def stop_vllm() -> dict[str, Any]:
    remote_script = """
set -euo pipefail
python3 - <<'PY'
import json
import os
import signal
import subprocess
from pathlib import Path

pid_file = Path("/home/pto/lean/logs/vllm_8000.pid")
stopped_pids = []
vllm_exec = "__VLLM_EXEC__"
vllm_python_launch = "__VLLM_PYTHON_LAUNCH__"
served_model_name = "__SERVED_MODEL_NAME__"

def list_matching_vllm_pids() -> list[int]:
    ps = subprocess.run(
        ["ps", "-e", "-o", "pid=", "-o", "args="],
        capture_output=True,
        text=True,
        check=True,
    )
    matched: list[int] = []
    for line in ps.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if vllm_exec + " serve" not in command and vllm_python_launch + " serve" not in command:
            continue
        if "--port 8000" not in command:
            continue
        if f"--served-model-name {served_model_name}" not in command:
            continue
        matched.append(int(pid_text))
    return matched

def list_descendants(root_pid: int) -> list[int]:
    ps = subprocess.run(
        ["ps", "-e", "-o", "pid=", "-o", "ppid="],
        capture_output=True,
        text=True,
        check=True,
    )
    children_by_parent: dict[int, list[int]] = {}
    for line in ps.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        child_pid, parent_pid = (int(part) for part in parts)
        children_by_parent.setdefault(parent_pid, []).append(child_pid)

    descendants: list[int] = []
    stack = list(children_by_parent.get(root_pid, ()))
    while stack:
        child_pid = stack.pop()
        descendants.append(child_pid)
        stack.extend(children_by_parent.get(child_pid, ()))
    return descendants

if pid_file.exists():
    raw = pid_file.read_text().strip()
    if raw:
        try:
            pid = int(raw)
            for child_pid in reversed(list_descendants(pid)):
                try:
                    os.kill(child_pid, signal.SIGTERM)
                    stopped_pids.append(child_pid)
                except ProcessLookupError:
                    pass
            try:
                os.kill(pid, signal.SIGTERM)
                stopped_pids.append(pid)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        except OSError:
            pass
    pid_file.unlink(missing_ok=True)

for pid in list_matching_vllm_pids():
    if pid in stopped_pids:
        continue
    try:
        for child_pid in reversed(list_descendants(pid)):
            try:
                os.kill(child_pid, signal.SIGTERM)
                stopped_pids.append(child_pid)
            except ProcessLookupError:
                pass
        os.kill(pid, signal.SIGTERM)
        stopped_pids.append(pid)
    except ProcessLookupError:
        pass
subprocess.run(["sleep", "2"], check=False)
print(json.dumps({"status": "stopped", "stopped_pids": stopped_pids}))
PY
""".strip()
    remote_script = remote_script.replace("__VLLM_EXEC__", DEFAULT_VLLM_EXEC).replace(
        "__VLLM_PYTHON_LAUNCH__",
        DEFAULT_VLLM_PYTHON_LAUNCH,
    ).replace(
        "__SERVED_MODEL_NAME__",
        DEFAULT_MODEL_NAME,
    )
    result = _run_remote_bash(remote_script)
    _raise_on_failure(result, "vLLM stop")
    return json.loads(result.stdout)


def run_vllm_benchmark(
    *,
    prompt: str,
    profile: str = DEFAULT_PROFILE,
    max_new_tokens: int | None = None,
    baseline_mode: str = DEFAULT_VLLM_BASELINE_MODE,
    baseline_runs: int = DEFAULT_VLLM_BASELINE_RUNS,
) -> dict[str, Any]:
    normalized_mode = baseline_mode.strip().lower()
    if normalized_mode not in {"plain", "best"}:
        raise ValueError("baseline_mode must be one of: plain, best")
    run_count = 1 if normalized_mode == "plain" else max(1, int(baseline_runs))

    env = {
        "PROFILE": profile,
        "SYSTEM_LABEL": "vllm",
        "VARIANT_LABEL": f"{normalized_mode}_openai",
        "MODEL_NAME": DEFAULT_MODEL_NAME,
        "BASE_URL": "http://127.0.0.1:8000",
        "PROMPT_OVERRIDE": prompt,
        "TEMPERATURE": str(DEFAULT_SAMPLING_TEMPERATURE),
        "IGNORE_EOS": "1" if DEFAULT_IGNORE_EOS else "0",
        "SKIP_REMOTE_SYNC": "1",
    }
    if max_new_tokens is not None:
        env["MAX_NEW_TOKENS_OVERRIDE"] = str(max_new_tokens)

    runs: list[dict[str, Any]] = []
    for _ in range(run_count):
        result = _run_shell_script("remote_openai_backend_benchmark.sh", env=env)
        _raise_on_failure(result, "vLLM benchmark")
        runs.append(_extract_last_json(result.stdout))
    if not runs:
        raise RuntimeError("vLLM baseline benchmark produced no results")

    def _score(payload: dict[str, Any]) -> float:
        value = payload.get("generated_tokens_per_second")
        if value is None:
            return float("-inf")
        return float(value)

    selected = max(runs, key=_score) if normalized_mode == "best" else runs[0]
    selected["baseline_mode"] = normalized_mode
    selected["baseline_runs"] = run_count
    selected["baseline_candidates_tokens_per_second"] = [run.get("generated_tokens_per_second") for run in runs]
    return selected


def run_leanstack_benchmark(
    *,
    prompt: str,
    profile: str = DEFAULT_PROFILE,
    max_new_tokens: int | None = None,
    runtime_mode: str = DEFAULT_RUNTIME_MODE,
    resident_requests: int = 3,
    warmup_requests: int = 1,
) -> dict[str, Any]:
    env = {
        "MODEL_ID": DEFAULT_MODEL_ID,
        "PACK_DIR": DEFAULT_PACK_DIR,
        "STRICT_PACKED": "1",
        "STRICT_CONTRACT": "1",
        "PROFILE": profile,
        "RUNTIME_MODE": runtime_mode,
        "NUM_LAYERS": "0",
        "DTYPE": DEFAULT_DTYPE,
        "DEVICE": DEFAULT_DEVICE,
        "SPECULATIVE": "0",
        "PROMPT_OVERRIDE": prompt,
        "PROMPT_FORMAT_OVERRIDE": "raw",
        "IGNORE_EOS": "1" if DEFAULT_IGNORE_EOS else "0",
        "RESIDENT_REQUESTS": str(resident_requests),
        "WARMUP_REQUESTS": str(warmup_requests),
        "SKIP_REMOTE_SYNC": "1",
    }
    if max_new_tokens is not None:
        env["MAX_NEW_TOKENS_OVERRIDE"] = str(max_new_tokens)
    result = _run_shell_script("remote_leanstack_benchmark.sh", env=env)
    _raise_on_failure(result, "leanstack benchmark")
    return _extract_last_json(result.stdout)


def build_comparison_payload(
    *,
    prompt: str,
    profile: str = DEFAULT_PROFILE,
    max_new_tokens: int | None = None,
    vllm_baseline_mode: str = DEFAULT_VLLM_BASELINE_MODE,
    vllm_baseline_runs: int = DEFAULT_VLLM_BASELINE_RUNS,
) -> dict[str, Any]:
    if profile != DEFAULT_PROFILE:
        raise RuntimeError(
            f"official comparison is locked to profile={DEFAULT_PROFILE}; got profile={profile}"
        )
    status = check_remote_status()
    if not status.get("download_complete"):
        raise RuntimeError("Qwen3-4B-Base checkpoint is not fully downloaded on the remote machine yet.")
    if not status.get("pack_ready"):
        raise RuntimeError(f"leanpack artifact is not ready at {DEFAULT_PACK_DIR}.")
    vllm_status = ensure_vllm_ready()
    vllm = run_vllm_benchmark(
        prompt=prompt,
        profile=profile,
        max_new_tokens=max_new_tokens,
        baseline_mode=vllm_baseline_mode,
        baseline_runs=vllm_baseline_runs,
    )
    stop_status = stop_vllm()
    leanstack = run_leanstack_benchmark(prompt=prompt, profile=profile, max_new_tokens=max_new_tokens)
    fairness = validate_comparison_fairness(
        vllm=vllm,
        leanstack=leanstack,
        expected_max_new_tokens=max_new_tokens,
    )

    vllm_tps = vllm.get("generated_tokens_per_second")
    leanstack_tps = (leanstack.get("throughput") or {}).get("runtime_tokens_per_second")
    vllm_ttft = vllm.get("ttft_seconds")
    leanstack_prefill = (leanstack.get("timings") or {}).get("prefill_seconds")

    return {
        "status": status,
        "official_contract": {
            "id": OFFICIAL_CONTRACT_ID,
            "model_id": DEFAULT_MODEL_ID,
            "profile": DEFAULT_PROFILE,
            "runtime_mode": DEFAULT_RUNTIME_MODE,
            "dtype": DEFAULT_DTYPE,
            "device": DEFAULT_DEVICE,
            "pack_dir": DEFAULT_PACK_DIR,
            "hardware": OFFICIAL_HARDWARE,
            "strict_packed": True,
            "strict_contract": True,
            "vllm_baseline_mode": vllm_baseline_mode,
            "vllm_baseline_runs": vllm_baseline_runs,
            "decode_policy": DEFAULT_DECODE_POLICY,
            "temperature": DEFAULT_SAMPLING_TEMPERATURE,
            "ignore_eos": DEFAULT_IGNORE_EOS,
        },
        "vllm_status": vllm_status,
        "vllm_stop_status": stop_status,
        "fairness_gate": fairness,
        "profile": profile,
        "prompt": prompt,
        "vllm": vllm,
        "leanstack": leanstack,
        "delta": {
            "runtime_tokens_per_second_ratio": (leanstack_tps / vllm_tps) if leanstack_tps and vllm_tps else None,
            "prefill_to_vllm_ttft_ratio": (leanstack_prefill / vllm_ttft) if leanstack_prefill and vllm_ttft else None,
        },
    }
