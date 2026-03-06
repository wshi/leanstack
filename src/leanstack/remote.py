from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .config import RemoteEndpoint


def parse_remote_script(path: Path) -> RemoteEndpoint:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"remote script is empty: {path}")

    tokens = shlex.split(raw)
    if not tokens or tokens[0] != "ssh":
        raise ValueError(f"unsupported remote script format: {raw}")

    user_host = ""
    port = 22
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-p":
            index += 1
            port = int(tokens[index])
        elif token.startswith("-"):
            pass
        elif not user_host:
            user_host = token
        index += 1

    if not user_host:
        raise ValueError(f"could not find remote host in: {raw}")

    return RemoteEndpoint(user_host=user_host, port=port, ssh_command=tuple(tokens))


def ssh_prefix(endpoint: RemoteEndpoint) -> list[str]:
    if endpoint.ssh_command:
        return list(endpoint.ssh_command)
    return ["ssh", endpoint.user_host, "-p", str(endpoint.port)]


def run_remote_bash(endpoint: RemoteEndpoint, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_prefix(endpoint) + [f"bash -lc {shlex.quote(script)}"],
        check=True,
        capture_output=True,
        text=True,
    )


def default_remote_script() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "remote.sh"
