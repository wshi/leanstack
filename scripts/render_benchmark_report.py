#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from leanstack.benchmark import load_benchmark_result, render_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a normalized benchmark table from JSON result files.")
    parser.add_argument("inputs", nargs="+", help="Input JSON result files.")
    parser.add_argument("--output", help="Optional markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [load_benchmark_result(Path(path)) for path in args.inputs]
    report = render_benchmark_report(results)
    print(report)
    if args.output:
        Path(args.output).write_text(f"{report}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
