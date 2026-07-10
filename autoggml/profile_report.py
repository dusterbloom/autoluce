"""Summarize rocprofv3 captures produced by remote experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoggml.bench.profiling import summarize_rocprofv3_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a rocprofv3 kernel CSV")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_rocprofv3_csv(args.capture)
    payload = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
