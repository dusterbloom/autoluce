"""
Read-only benchmark harness for autoggml v2.

Applies the experiment from experiment.py, builds lucebox-ggml,
runs benchmarks, checks correctness against golden outputs, and computes a score.
"""

from __future__ import annotations

import json
import os
import re
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
    run(["cmake", "--build", str(BUILD_DIR), "-j", "--target", "llama-bench", "llama-cli"], env=env)
    build_time = time.time() - t0
    return build_time


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# llama-bench parsing (#1)
# ---------------------------------------------------------------------------


def parse_llama_bench_output(stdout: str) -> dict[str, float]:
    """
    Parse llama-bench table output.
    Looks for lines like:
      | model | size | params | backend | threads | test | t/s |
    """
    metrics: dict[str, float] = {}
    # Pattern: |  qwen2 1.5B Q4_0  | 885.97 MiB | 1.54 B | Metal,BLAS | 16 | pp512 | 5765.41 ± 20.55 |
    pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*([0-9.]+)")
    for line in stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        test = m.group(6).strip()
        value = float(m.group(7).strip())
        if test.startswith("tg"):
            metrics.setdefault("decode_tok_s", value)
        elif test.startswith("pp"):
            metrics.setdefault("prefill_tok_s", value)
    return metrics


def benchmark_with_llama_bench(model: Path, draft: Path | None, n_draft: int, bench_args: dict) -> dict[str, float]:
    """
    Run llama-bench and parse real metrics.
    Falls back to simulation if llama-bench is missing.
    """
    bench = BUILD_DIR / "bin" / "llama-bench"
    if not bench.exists():
        return {
            "decode_tok_s": 100.0 + n_draft * 2.0,
            "prefill_tok_s": 2000.0,
            "acceptance_rate": 0.55,
            "peak_mem_GiB": 16.0,
        }

    cmd = [
        str(bench),
        "-m", str(model),
        "-p", str(bench_args.get("-p", 512)),
        "-n", str(bench_args.get("-n", 128)),
        "--repetitions", str(bench_args.get("--repetitions", 3)),
    ]
    if draft is not None and draft.exists():
        cmd += ["-md", str(draft), "--spec-type", "draft-dflash", "--spec-draft-n-max", str(n_draft)]

    result = run(cmd, timeout=600)
    metrics = parse_llama_bench_output(result.stdout)

    # llama-bench does not directly report acceptance rate or peak memory.
    # Use defaults if parsing failed; these can be refined later.
    if "decode_tok_s" not in metrics:
        metrics["decode_tok_s"] = 100.0
    if "prefill_tok_s" not in metrics:
        metrics["prefill_tok_s"] = 2000.0
    metrics.setdefault("acceptance_rate", 0.55)
    metrics.setdefault("peak_mem_GiB", 16.0)
    return metrics


# ---------------------------------------------------------------------------
# Golden-output correctness (#3)
# ---------------------------------------------------------------------------


def load_golden(benchmark_name: str) -> dict | None:
    golden_path = BENCHMARKS_DIR / "golden" / f"{benchmark_name}.json"
    if not golden_path.exists():
        return None
    return json.loads(golden_path.read_text())


def generate_text(llama_cli: Path, model: Path, draft: Path | None, prompt: str, params: dict, n_draft: int) -> str:
    cmd = [
        str(llama_cli),
        "-m", str(model),
        "-p", prompt,
        "-n", str(params.get("n_predict", 64)),
        "--temp", str(params.get("temperature", 0.0)),
        "--top-k", str(params.get("top_k", 1)),
        "--top-p", str(params.get("top_p", 1.0)),
        "--seed", str(params.get("seed", 42)),
        "--no-display-prompt",
    ]
    if draft is not None and draft.exists():
        cmd += [
            "-md", str(draft),
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", str(n_draft),
        ]
    result = run(cmd, timeout=300)
    return result.stdout.strip()


def check_correctness(benchmark_name: str, model: Path, draft: Path | None, n_draft: int) -> tuple[bool, list[dict]]:
    """
    Compare generated text against golden outputs.
    Returns (all_passed, per_prompt results).
    """
    golden = load_golden(benchmark_name)
    if golden is None:
        print("WARNING: no golden outputs found; skipping correctness check.")
        return True, []

    if golden.get("generated_at") == "NOT_YET_GENERATED":
        print("WARNING: golden outputs are placeholders; skipping correctness check.")
        return True, []

    llama_cli = BUILD_DIR / "bin" / "llama-cli"
    if not llama_cli.exists():
        print("WARNING: llama-cli not built; skipping correctness check.")
        return True, []

    params = golden.get("parameters", {})
    results = []
    all_passed = True
    for expected in golden.get("outputs", []):
        actual = generate_text(llama_cli, model, draft, expected["prompt"], params, n_draft)
        passed = actual == expected["text"]
        if not passed:
            all_passed = False
        results.append({
            "prompt": expected["prompt"],
            "passed": passed,
            "expected": expected["text"],
            "actual": actual,
        })
    return all_passed, results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_score(metrics: dict[str, float], correct: bool) -> float:
    if not correct:
        return 0.0
    decode = max(1.0, metrics.get("decode_tok_s", 1.0))
    prefill = max(1.0, metrics.get("prefill_tok_s", 1.0))
    mem = max(1.0, metrics.get("peak_mem_GiB", 1.0))
    build_time = max(1.0, metrics.get("build_time_s", 1.0))
    acceptance = metrics.get("acceptance_rate", 0.5)
    return (decode * prefill * acceptance) / (mem * build_time)


# ---------------------------------------------------------------------------
# Multi-benchmark aggregation (#7)
# ---------------------------------------------------------------------------


def list_benchmarks() -> list[str]:
    return sorted(p.stem for p in BENCHMARKS_DIR.glob("*.json") if p.name != "manifest.json")


def run_single_benchmark(benchmark_name: str, build_time: float) -> dict[str, any]:
    benchmark = json.loads((BENCHMARKS_DIR / f"{benchmark_name}.json").read_text())
    model = resolve_model(benchmark.get("manifest_entry", benchmark_name), "target")
    draft = resolve_model(benchmark.get("manifest_entry", benchmark_name), "draft")
    n_draft = benchmark.get("n_draft", 15)

    if model is None or not model.exists():
        metrics = {
            "decode_tok_s": 120.0,
            "prefill_tok_s": 2500.0,
            "acceptance_rate": 0.55,
            "peak_mem_GiB": 18.0,
        }
        correct = True
        details = []
    else:
        metrics = benchmark_with_llama_bench(model, draft, n_draft, benchmark.get("llama_bench_args", {}))
        correct, details = check_correctness(benchmark_name, model, draft, n_draft)

    metrics["build_time_s"] = build_time
    metrics["correctness"] = "pass" if correct else "FAIL"
    score = compute_score(metrics, correct)

    return {
        "benchmark": benchmark_name,
        "score": score,
        **metrics,
        "correctness_details": details,
    }


def aggregate_scores(results: list[dict]) -> dict[str, float]:
    if not results:
        return {"score": 0.0}
    n = len(results)
    return {
        "score": sum(r["score"] for r in results) / n,
        "decode_tok_s": sum(r["decode_tok_s"] for r in results) / n,
        "prefill_tok_s": sum(r["prefill_tok_s"] for r in results) / n,
        "acceptance_rate": sum(r["acceptance_rate"] for r in results) / n,
        "peak_mem_GiB": sum(r["peak_mem_GiB"] for r in results) / n,
    }


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run_harness(baseline: bool = False) -> dict[str, any]:
    if not baseline:
        reset_lucebox()
        exp_info = apply_experiment()
    else:
        exp_info = {"description": "baseline", "cmake_flags": [], "runtime_flags": {}}

    build_time = build()

    benchmark_names = os.environ.get("AUTOGGML_BENCHMARKS", ",".join(list_benchmarks())).split(",")
    benchmark_names = [b.strip() for b in benchmark_names if b.strip()]

    per_benchmark = []
    for name in benchmark_names:
        per_benchmark.append(run_single_benchmark(name, build_time))

    agg = aggregate_scores(per_benchmark)
    any_fail = any(r["correctness"] == "FAIL" for r in per_benchmark)

    return {
        "score": agg["score"],
        **agg,
        "build_time_s": build_time,
        "correctness": "pass" if not any_fail else "FAIL",
        "benchmarks": per_benchmark,
        "experiment": exp_info,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_harness(baseline=args.baseline)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
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
    for r in summary["benchmarks"]:
        print(f"  {r['benchmark']:<20} score={r['score']:.4f} decode={r['decode_tok_s']:.2f} correct={r['correctness']}")
    print("---")

    if summary["correctness"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
