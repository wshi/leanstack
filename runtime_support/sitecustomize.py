from __future__ import annotations

import os
import sys
import sysconfig


_ORIGINAL_GET_PATHS = sysconfig.get_paths


def _resolve_include_dir() -> str | None:
    dev_root = os.environ.get("LEANSTACK_PYTHON_DEV_ROOT")
    if not dev_root:
        return None
    include_dir = os.path.join(dev_root, "usr", "include", f"python{sys.version_info.major}.{sys.version_info.minor}")
    if os.path.isfile(os.path.join(include_dir, "Python.h")):
        _ensure_arch_include_link(include_dir, dev_root)
        return include_dir
    return None


def _ensure_arch_include_link(include_dir: str, dev_root: str) -> None:
    arch_root = os.path.join(dev_root, "usr", "include", "aarch64-linux-gnu")
    if not os.path.isdir(arch_root):
        return
    link_path = os.path.join(include_dir, "aarch64-linux-gnu")
    if os.path.exists(link_path):
        return
    try:
        os.symlink(os.path.relpath(arch_root, include_dir), link_path)
    except OSError:
        return


def _patched_get_paths(*args, **kwargs):
    paths = dict(_ORIGINAL_GET_PATHS(*args, **kwargs))
    include_dir = _resolve_include_dir()
    if include_dir:
        paths["include"] = include_dir
    return paths


sysconfig.get_paths = _patched_get_paths
