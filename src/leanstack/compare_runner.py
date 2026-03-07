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
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_DIR = "/home/pto/lean/models/Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_PATH_FILE = "/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path"
DEFAULT_PROFILE = "decode_64_256"
DEFAULT_MODEL_NAME = "qwen3-1.7b-base"
DEFAULT_VLLM_VENV = "/home/pto/lean/venv-vllm-cu128"
DEFAULT_PYTHON_DEV_ROOT = "/home/pto/lean/tmp/pydev_probe/extracted"
DEFAULT_VLLM_EXEC = f"{DEFAULT_VLLM_VENV}/bin/vllm"
DEFAULT_VLLM_PYTHON_LAUNCH = f"{DEFAULT_VLLM_VENV}/bin/python3 {DEFAULT_VLLM_EXEC}"


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

model_dir = Path("/home/pto/lean/models/Qwen/Qwen3-1.7B-Base")
path_file = Path("/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path")
weight_file = model_dir / "model.safetensors"
fetch = subprocess.run(["pgrep", "-xaf", r"python3 .*fetch_modelscope_snapshot.py"], capture_output=True, text=True)
vllm_ready = subprocess.run(
    ["bash", "-lc", "curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1"],
    capture_output=True,
)

size_bytes = 0
if model_dir.exists():
    for path in model_dir.rglob("*"):
        if path.is_file():
            size_bytes += path.stat().st_size

payload = {
    "path_file_exists": path_file.exists(),
    "path_file": path_file.read_text().strip() if path_file.exists() else None,
    "model_dir_exists": model_dir.exists(),
    "model_files": sorted(path.name for path in model_dir.glob("*") if path.is_file()) if model_dir.exists() else [],
    "model_size_bytes": size_bytes,
    "weight_file_exists": weight_file.exists(),
    "weight_size_bytes": weight_file.stat().st_size if weight_file.exists() else 0,
    "fetch_processes": [line for line in fetch.stdout.splitlines() if line.strip()],
    "vllm_ready": vllm_ready.returncode == 0,
}
payload["download_complete"] = payload["weight_file_exists"]
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
        if "/home/pto/lean/models/Qwen/Qwen3-1___7B-Base" not in command:
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
    )
    result = _run_remote_bash(remote_script)
    _raise_on_failure(result, "vLLM stop")
    return json.loads(result.stdout)


def run_vllm_benchmark(
    *,
    prompt: str,
    profile: str = DEFAULT_PROFILE,
    max_new_tokens: int | None = None,
    warmup_requests: int = 1,
) -> dict[str, Any]:
    env = {
        "PROFILE": profile,
        "SYSTEM_LABEL": "vllm",
        "VARIANT_LABEL": "openai",
        "MODEL_NAME": DEFAULT_MODEL_NAME,
        "BASE_URL": "http://127.0.0.1:8000",
        "PROMPT_OVERRIDE": prompt,
        "SKIP_REMOTE_SYNC": "1",
    }
    if max_new_tokens is not None:
        env["MAX_NEW_TOKENS_OVERRIDE"] = str(max_new_tokens)
    for _ in range(max(0, warmup_requests)):
        warmup_result = _run_shell_script("remote_openai_backend_benchmark.sh", env=env)
        _raise_on_failure(warmup_result, "vLLM warmup benchmark")
    result = _run_shell_script("remote_openai_backend_benchmark.sh", env=env)
    _raise_on_failure(result, "vLLM benchmark")
    return _extract_last_json(result.stdout)


def run_leanstack_benchmark(
    *,
    prompt: str,
    profile: str = DEFAULT_PROFILE,
    max_new_tokens: int | None = None,
    runtime_mode: str = "semantic",
    resident_requests: int = 3,
    warmup_requests: int = 1,
) -> dict[str, Any]:
    env = {
        "MODEL_ID": DEFAULT_MODEL_ID,
        "PROFILE": profile,
        "RUNTIME_MODE": runtime_mode,
        "NUM_LAYERS": "0",
        "PROMPT_OVERRIDE": prompt,
        "PROMPT_FORMAT_OVERRIDE": "raw",
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
) -> dict[str, Any]:
    status = check_remote_status()
    if not status.get("download_complete"):
        raise RuntimeError("Qwen3-1.7B-Base checkpoint is not fully downloaded on the remote machine yet.")
    vllm_status = ensure_vllm_ready()
    vllm = run_vllm_benchmark(prompt=prompt, profile=profile, max_new_tokens=max_new_tokens)
    stop_status = stop_vllm()
    leanstack = run_leanstack_benchmark(prompt=prompt, profile=profile, max_new_tokens=max_new_tokens)

    vllm_tps = vllm.get("generated_tokens_per_second")
    leanstack_tps = (leanstack.get("throughput") or {}).get("runtime_tokens_per_second")
    vllm_ttft = vllm.get("ttft_seconds")
    leanstack_prefill = (leanstack.get("timings") or {}).get("prefill_seconds")

    return {
        "status": status,
        "vllm_status": vllm_status,
        "vllm_stop_status": stop_status,
        "profile": profile,
        "prompt": prompt,
        "vllm": vllm,
        "leanstack": leanstack,
        "delta": {
            "runtime_tokens_per_second_ratio": (leanstack_tps / vllm_tps) if leanstack_tps and vllm_tps else None,
            "prefill_to_vllm_ttft_ratio": (leanstack_prefill / vllm_ttft) if leanstack_prefill and vllm_ttft else None,
        },
    }
