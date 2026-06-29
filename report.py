"""
Read-only result aggregation and diff tool.

Usage:
    uv run report.py results/20260101T000000 results/20260102T000000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_summary(dir_path: Path) -> dict:
    path = dir_path / "baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"No baseline.json in {dir_path}")
    return json.loads(path.read_text())


def print_diff(a: dict, b: dict) -> None:
    keys = ["score", "decode_tok_s", "prefill_tok_s", "acceptance_rate", "peak_mem_GiB", "build_time_s"]
    print(f"{'metric':<20} {'run_a':>12} {'run_b':>12} {'delta':>12} {'pct':>10}")
    print("-" * 70)
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        delta = vb - va
        pct = (delta / va * 100.0) if va != 0 else 0.0
        print(f"{k:<20} {va:>12.4f} {vb:>12.4f} {delta:>12.4f} {pct:>9.2f}%")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <run_a_dir> <run_b_dir>")
        sys.exit(1)
    a_dir = Path(sys.argv[1])
    b_dir = Path(sys.argv[2])
    a = load_summary(a_dir)
    b = load_summary(b_dir)
    print_diff(a, b)


if __name__ == "__main__":
    main()
