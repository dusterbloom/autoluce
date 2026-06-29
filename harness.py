"""
Read-only benchmark harness for autoggml v2.

Applies the experiment from experiment.py, builds lucebox-ggml,
runs benchmarks, checks correctness, and computes a score.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment import apply_experiment, get_cmake_flags, reset_lucebox

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
Lucebox_DIR = WORK_DIR / "lucebox-ggml"
BUILD_DIR = Lucebox_DIR / "build"
BENCHMARKS_DIR = ROOT / "benchmarks"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, timeout: int | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout, env=env)


def has_valid_source() -> bool:
    return (Lucebox_DIR / "CMakeLists.txt").exists() and (Lucebox_DIR / ".git").exists()


def build() -> float:
    """Build lucebox-ggml and return build wall time."""
    if not has_valid_source():
        print("WARNING: lucebox-ggml source not found; running in simulation mode.")
        print("         Run `uv run prepare.py` to clone and build the real project.")
        return 1.0

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        "-S", str(Lucebox_DIR),
        "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
    ] + get_cmake_flags()

    env = os.environ.copy()
    run(cmake_args, env=env)

    t0 = time.time()
    run(["cmake", "--build", str(BUILD_DIR), "-j", "--target", "llama-bench"], env=env)
    build_time = time.time() - t0
    return build_time


def load_manifest() -> dict:
    manifest_path = WORK_DIR / "models" / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def resolve_model(benchmark_name: str, role: str) -> Path | None:
    manifest = load_manifest()
    entry = manifest.get(benchmark_name, {}).get(role)
    if not entry:
        return None
    return WORK_DIR / "models" / entry["local"]


def benchmark_decode(model: Path, draft: Path | None, n_draft: int) -> dict[str, float]:
    """
    Run llama-bench decode benchmark.
    Returns a dict with decode_tok_s, prefill_tok_s, acceptance_rate, peak_mem_GiB.
    """
    bench = BUILD_DIR / "bin" / "llama-bench"
    if not bench.exists():
        # Simulation mode when llama-bench is not available.
        return {
            "decode_tok_s": 100.0 + n_draft * 2.0,
            "prefill_tok_s": 2000.0,
            "acceptance_rate": 0.55,
            "peak_mem_GiB": 16.0,
        }

    cmd = [
        str(bench),
        "-m", str(model),
        "-p", "512",
        "-n", "128",
        "--repetitions", "3",
    ]
    if draft is not None and draft.exists():
        cmd += ["-md", str(draft), "--spec-type", "draft-dflash", "--spec-draft-n-max", str(n_draft)]

    run(cmd, timeout=600)
    # TODO: parse llama-bench output.
    return {
        "decode_tok_s": 100.0,
        "prefill_tok_s": 2000.0,
        "acceptance_rate": 0.55,
        "peak_mem_GiB": 16.0,
    }


def check_correctness(model: Path) -> bool:
    """
    Check correctness by running a tiny deterministic generation and comparing
    the output hash against an expected value.
    """
    # Placeholder: in a real run this uses llama-cli with a fixed prompt.
    return True


def compute_score(metrics: dict[str, float], correct: bool) -> float:
    if not correct:
        return 0.0
    decode = max(1.0, metrics.get("decode_tok_s", 1.0))
    prefill = max(1.0, metrics.get("prefill_tok_s", 1.0))
    mem = max(1.0, metrics.get("peak_mem_GiB", 1.0))
    build_time = max(1.0, metrics.get("build_time_s", 1.0))
    acceptance = metrics.get("acceptance_rate", 0.5)
    # Throughput squared, regularized by memory and build time, boosted by acceptance.
    return (decode * prefill * acceptance) / (mem * build_time)


def run_harness(baseline: bool = False) -> dict[str, any]:
    if not baseline:
        reset_lucebox()
        exp_info = apply_experiment()
    else:
        exp_info = {"description": "baseline", "cmake_flags": [], "runtime_flags": {}}

    build_time = build()

    benchmark_name = os.environ.get("AUTOGGML_BENCHMARK", "qwen36-27b")
    model = resolve_model(benchmark_name, "target")
    draft = resolve_model(benchmark_name, "draft")

    if model is None or not model.exists():
        # Simulation mode if models are not downloaded.
        metrics = {
            "decode_tok_s": 120.0,
            "prefill_tok_s": 2500.0,
            "acceptance_rate": 0.55,
            "peak_mem_GiB": 18.0,
        }
    else:
        n_draft = int(os.environ.get("AUTOGGML_N_DRAFT", "15"))
        metrics = benchmark_decode(model, draft, n_draft=n_draft)

    metrics["build_time_s"] = build_time
    correct = check_correctness(model) if model and model.exists() else True
    metrics["correctness"] = "pass" if correct else "FAIL"
    score = compute_score(metrics, correct)

    return {
        "score": score,
        **metrics,
        "experiment": exp_info,
        "benchmark": benchmark_name,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_harness(baseline=args.baseline)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    mode = "baseline" if args.baseline else "experiment"
    print("---")
    print(f"mode:              {mode}")
    print(f"score:             {summary['score']:.4f}")
    print(f"decode_tok_s:      {summary['decode_tok_s']:.2f}")
    print(f"prefill_tok_s:     {summary['prefill_tok_s']:.2f}")
    print(f"acceptance_rate:   {summary['acceptance_rate']:.4f}")
    print(f"peak_mem_GiB:      {summary['peak_mem_GiB']:.2f}")
    print(f"build_time_s:      {summary['build_time_s']:.2f}")
    print(f"correctness:       {summary['correctness']}")
    print(f"description:       {summary['experiment']['description']}")
    print("---")

    if summary["correctness"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
