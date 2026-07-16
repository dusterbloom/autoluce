"""
Generate golden outputs for correctness tests.

Usage:
    uv run scripts/generate_golden.py --benchmark qwen36-27b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoluce.bench.profiling import detect_backend  # noqa: E402
from autoluce.runtime.dflash_http import DflashHttpRuntime  # noqa: E402
from autoluce.source_layout import SourceLayout  # noqa: E402

WORK_DIR = ROOT / "work"
MODELS_DIR = WORK_DIR / "models"
BENCHMARKS_DIR = ROOT / "benchmarks"
GOLDEN_DIR = Path(os.environ.get("AUTOLUCE_GOLDEN_DIR", str(BENCHMARKS_DIR / "golden")))


def load_prompts(benchmark: dict) -> list[str]:
    if benchmark.get("prompts"):
        return [str(prompt) for prompt in benchmark["prompts"]]
    prompts_file = BENCHMARKS_DIR / "prompts.txt"
    return [p.strip() for p in prompts_file.read_text().split("---") if p.strip()]


def load_benchmark(name: str) -> dict:
    return json.loads((BENCHMARKS_DIR / f"{name}.json").read_text())


def load_manifest() -> dict:
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def resolve_model(benchmark_name: str, role: str) -> Path | None:
    manifest = load_manifest()
    entry = manifest.get(benchmark_name, {}).get(role)
    if not entry:
        return None
    if entry.get("path"):
        return Path(entry["path"]).expanduser()
    files = entry.get("files")
    if files:
        first = Path(files[0]).expanduser()
        return first if first.is_absolute() else MODELS_DIR / first
    return MODELS_DIR / entry["local"]


def resolve_benchmark_models(benchmark: dict) -> tuple[Path | None, Path | None]:
    """Resolve the target and honor a target-only benchmark during quality freeze."""

    entry = benchmark.get("manifest_entry", benchmark.get("name"))
    target = resolve_model(entry, "target")
    draft = None if benchmark.get("spec_type") == "target-only" else resolve_model(entry, "draft")
    return target, draft


def generate_one(client, prompt: str, params: dict) -> dict:
    return {
        "prompt": prompt,
        "text": client.complete(prompt, params).text,
        "tokens": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="qwen36-27b")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    layout = SourceLayout.resolve()
    layout.require_capability("product-quality-exact")
    backend = detect_backend()
    server = layout.binary("dflash_server", backend)
    if not server.exists():
        print(f"ERROR: {server} not found. Run `uv run autoluce setup` first.")
        return 1

    benchmark = load_benchmark(args.benchmark)
    model, draft = resolve_benchmark_models(benchmark)

    if model is None or not model.exists():
        print(f"ERROR: target model not found: {model}")
        return 1

    golden_path = GOLDEN_DIR / f"{args.benchmark}.json"
    if golden_path.exists() and not args.overwrite:
        print(f"Golden output already exists at {golden_path}. Use --overwrite to regenerate.")
        return 0

    prompts = load_prompts(benchmark)
    params = benchmark.get("parameters", {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "n_predict": 64, "seed": 42})
    params["n_draft"] = benchmark.get("n_draft", 15)

    max_context = max([int(value) for value in benchmark.get("contexts", [])] or [int(benchmark.get("max_context", 8192))])
    runtime = DflashHttpRuntime(layout, backend, model, draft, {})
    outputs = []
    with runtime.session(max_context) as (client, _server, _telemetry):
        for prompt in prompts:
            print(f"Generating golden output for: {prompt[:60]}...")
            outputs.append(generate_one(client, prompt, params))

    golden = {
        "benchmark": args.benchmark,
        "model": benchmark.get("manifest_entry", args.benchmark),
        "draft": str(draft) if draft else None,
        "spec_type": benchmark.get("spec_type"),
        "n_draft": benchmark.get("n_draft"),
        "parameters": params,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(golden, indent=2))
    print(f"Golden outputs written to {golden_path}")


if __name__ == "__main__":
    raise SystemExit(main())
