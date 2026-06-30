"""
Generate golden outputs for correctness tests.

Usage:
    uv run scripts/generate_golden.py --benchmark qwen36-27b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import build_generation_command, extract_generated_text  # noqa: E402

WORK_DIR = ROOT / "work"
BUILD_DIR = WORK_DIR / "lucebox-ggml" / "build"
MODELS_DIR = WORK_DIR / "models"
BENCHMARKS_DIR = ROOT / "benchmarks"


def load_prompts() -> list[str]:
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
    return MODELS_DIR / entry["local"]


def generate_one(llama_cli: Path, model: Path, draft: Path | None, prompt: str, params: dict) -> dict:
    cmd = build_generation_command(llama_cli, model, draft, prompt, params, params.get("n_draft", 15))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {result.stderr}")

    return {
        "prompt": prompt,
        "text": extract_generated_text(result.stdout),
        "tokens": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="qwen36-27b")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    llama_cli = BUILD_DIR / "bin" / "llama-cli"
    if not llama_cli.exists():
        print(f"ERROR: {llama_cli} not found. Run `uv run prepare.py` first.")
        return 1

    benchmark = load_benchmark(args.benchmark)
    model = resolve_model(args.benchmark, "target")
    draft = resolve_model(args.benchmark, "draft")

    if model is None or not model.exists():
        print(f"ERROR: target model not found: {model}")
        return 1

    golden_path = BENCHMARKS_DIR / "golden" / f"{args.benchmark}.json"
    if golden_path.exists() and not args.overwrite:
        print(f"Golden output already exists at {golden_path}. Use --overwrite to regenerate.")
        return 0

    prompts = load_prompts()
    params = benchmark.get("parameters", {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "n_predict": 64, "seed": 42})
    params["n_draft"] = benchmark.get("n_draft", 15)

    outputs = []
    for prompt in prompts:
        print(f"Generating golden output for: {prompt[:60]}...")
        outputs.append(generate_one(llama_cli, model, draft, prompt, params))

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
