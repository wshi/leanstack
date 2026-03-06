from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model snapshot from ModelScope.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--allow-pattern", action="append")
    parser.add_argument("--path-file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "modelscope is not installed. Install it first or relay the model from the Mac."
        ) from exc

    resolved_path = snapshot_download(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
        allow_file_pattern=args.allow_pattern,
    )

    payload = {
        "model_id": args.model_id,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "revision": args.revision,
        "allow_pattern": args.allow_pattern,
        "resolved_path": resolved_path,
    }
    text = json.dumps(payload, indent=2)
    print(text)

    if args.path_file:
        Path(args.path_file).write_text(f"{resolved_path}\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
