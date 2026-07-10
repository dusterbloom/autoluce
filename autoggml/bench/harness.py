"""
Read-only benchmark harness for autoggml v2.

Applies the experiment from experiment.py, builds the pinned Lucebox product,
runs benchmarks, checks correctness against golden outputs, and computes a score.

Real mode (default) requires a prepared work/ tree and measures every metric;
missing prerequisites raise rather than degrade into fabricated numbers.
Use --simulate for plumbing tests without models or a build (results are clearly
labelled and never reflect a real measurement).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiment import apply_experiment, get_cmake_flags, get_runtime_flags, reset_lucebox
from autoggml.bench.kl import check_kl, kl_gate, resolve_kl_tau
from autoggml.bench.objective import check_constraints
from autoggml.bench.profiling import detect_backend, profile_command
from autoggml.bench.telemetry import TelemetryCollector
from autoggml.bench.uncertainty import propagate_score_stddev
from autoggml.source_layout import SourceLayout

from autoggml import ROOT
WORK_DIR = ROOT / "work"
BENCHMARKS_DIR = ROOT / "benchmarks"
STATE_DIR = Path(os.environ.get("AUTOGGML_STATE_DIR", str(WORK_DIR)))
GOLDEN_DIR = Path(os.environ.get("AUTOGGML_GOLDEN_DIR", str(BENCHMARKS_DIR / "golden")))
BASELINE_METRICS_PATH = STATE_DIR / "baseline_metrics.json"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, timeout: int | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout, env=env)


def has_valid_source() -> bool:
    layout = SourceLayout.resolve()
    return layout.checkout.exists() and layout.detect() == layout.manifest.layout


def build() -> float:
    """Build current Lucebox product targets and return wall time."""
    if not has_valid_source():
        raise RuntimeError("Lucebox product source not found; run `uv run autoggml setup` (or use --simulate)")
    from autoggml.prepare import build_commands

    layout = SourceLayout.resolve()
    backend = detect_backend()
    jobs = min(4, max(1, int(os.environ.get("AUTOGGML_BUILD_JOBS", "4"))))
    cmake_args, build_args = build_commands(layout, backend, jobs, use_ccache=bool(shutil.which("ccache")))
    cmake_args += get_cmake_flags()
    layout.build_dir(backend).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    run(cmake_args, env=env)
    t0 = time.time()
    run(build_args, env=env)
    return time.time() - t0


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
    if entry.get("path"):
        return Path(entry["path"]).expanduser()
    files = entry.get("files")
    if files:
        first = Path(files[0]).expanduser()
        return first if first.is_absolute() else WORK_DIR / "models" / first
    return WORK_DIR / "models" / entry["local"]


# ---------------------------------------------------------------------------
# llama-bench parsing
# ---------------------------------------------------------------------------


def parse_llama_bench_output(stdout: str) -> dict[str, float]:
    """
    Parse llama-bench table output into whatever throughput metrics it reports.
    Returns only fields that were actually parsed; callers decide what is required.
    """
    metrics: dict[str, float] = {}
    pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*([0-9.]+)(?:\s*±\s*([0-9.]+))?")
    for line in stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        test = m.group(6).strip()
        if test.startswith("tg"):
            name = "decode_tok_s"
        elif test.startswith("pp"):
            name = "prefill_tok_s"
        else:
            continue
        metrics.setdefault(name, float(m.group(7).strip()))
        if m.group(8) is not None:
            metrics.setdefault(f"{name}_stddev", float(m.group(8).strip()))
    return metrics


def parse_llama_bench_json(stdout: str) -> dict[str, float]:
    """Parse llama-bench JSON rows, including context-conditioned decode tests."""
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("could not parse llama-bench JSON output") from exc
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise RuntimeError("llama-bench JSON output must be an object or array")

    metrics: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        value = float(row.get("avg_ts", 0.0))
        sigma = float(row.get("stddev_ts", 0.0))
        if n_gen > 0 and n_prompt == 0:
            metrics.setdefault("decode_tok_s", value)
            metrics.setdefault("decode_tok_s_stddev", sigma)
        elif n_prompt > 0 and n_gen == 0:
            metrics.setdefault("prefill_tok_s", value)
            metrics.setdefault("prefill_tok_s_stddev", sigma)
    return metrics


def gnu_time_binary() -> str:
    for candidate in ("/usr/bin/time", shutil.which("time") or ""):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("GNU time (/usr/bin/time) not found; required for peak memory measurement")


def parse_peak_memory(time_stderr: str) -> float:
    """Parse peak resident set size in GiB from `time -v` stderr."""
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_stderr)
    if not m:
        raise RuntimeError("could not parse peak memory from `time -v` output")
    return int(m.group(1)) / (1024 * 1024)


def parse_acceptance_rate(stdout: str) -> float | None:
    """Best-effort speculative acceptance rate in [0, 1]; None when not reported."""
    m = re.search(r"acc(?:eptance)?[\s:=]+(\d\.\d+)", stdout, re.IGNORECASE)
    return float(m.group(1)) if m else None


def require_acceptance_rate(stdout: str, bench_name: str) -> float:
    """Speculative runs must report acceptance; measure or raise (no neutral fallback)."""
    acceptance = parse_acceptance_rate(stdout)
    if acceptance is None:
        raise RuntimeError(f"{bench_name} used a draft model but did not report acceptance_rate")
    return acceptance


def run_with_peak_memory(cmd: list[str], timeout: int | None) -> tuple[str, float, dict[str, float]]:
    """Run cmd under GNU time while collecting host/UMA telemetry."""
    collector = TelemetryCollector()
    collector.start()
    try:
        proc = run([gnu_time_binary(), "-v", *cmd], timeout=timeout)
    finally:
        telemetry = collector.stop()
    return proc.stdout, parse_peak_memory(proc.stderr), telemetry


def apply_runtime_flags(cmd: list[str], runtime_flags: dict[str, str]) -> list[str]:
    """Append experiment runtime flags to a llama.cpp command."""
    for key, value in runtime_flags.items():
        flag = f"--{key}"
        if flag not in cmd:
            cmd.extend([flag, str(value)])
        else:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                cmd[idx + 1] = str(value)
    return cmd


def benchmark_with_llama_bench(
    model: Path,
    draft: Path | None,
    n_draft: int,
    bench_args: dict,
    runtime_flags: dict[str, str],
    profile_path: str | None = None,
    context_depth: int | None = None,
) -> dict[str, float]:
    """Run llama-bench and parse real metrics. Raises if it cannot measure."""
    layout = SourceLayout.resolve()
    layout.require_capability("llama-tools")
    bench = layout.build_dir(detect_backend()) / "bin" / "llama-bench"
    if not bench.exists():
        raise RuntimeError(f"{bench} not found; run prepare.py (or use --simulate)")

    cmd = [
        str(bench),
        "-m", str(model),
        "-p", str(bench_args.get("-p", 512)),
        "-n", str(bench_args.get("-n", 128)),
        "--repetitions", str(bench_args.get("--repetitions", 3)),
        "-o", "json",
    ]
    if context_depth is not None:
        cmd += ["-d", str(context_depth)]
    speculative = draft is not None and draft.exists()
    if speculative:
        cmd += ["-md", str(draft), "--spec-type", "draft-dflash", "--spec-draft-n-max", str(n_draft)]
    cmd = apply_runtime_flags(cmd, runtime_flags)
    if profile_path is not None:
        cmd = profile_command(cmd, detect_backend(), profile_path)

    stdout, peak_mem, telemetry = run_with_peak_memory(cmd, timeout=3600)
    metrics = parse_llama_bench_json(stdout)
    for required in ("decode_tok_s", "prefill_tok_s"):
        if required not in metrics:
            raise RuntimeError(f"could not parse {required} from llama-bench output")

    if speculative:
        # Logged diagnostic only (not part of the score); measured or the run raises.
        metrics["acceptance_rate"] = require_acceptance_rate(stdout, bench.name)
    metrics["peak_mem_GiB"] = peak_mem
    metrics.update(telemetry)
    return metrics


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_context_metrics(cells: list[dict], primary_count: int = 2) -> dict:
    """Aggregate depth cells while keeping each cell available for constraints."""
    primary = cells[:primary_count]
    result = {
        "decode_tok_s": _geometric_mean([cell["decode_tok_s"] for cell in primary]),
        "decode_tok_s_stddev": math.sqrt(sum(cell.get("decode_tok_s_stddev", 0.0) ** 2 for cell in primary)) / max(1, len(primary)),
        "prefill_tok_s": _geometric_mean([cell["prefill_tok_s"] for cell in primary]),
        "prefill_tok_s_stddev": math.sqrt(sum(cell.get("prefill_tok_s_stddev", 0.0) ** 2 for cell in primary)) / max(1, len(primary)),
        "peak_mem_GiB": max(cell["peak_mem_GiB"] for cell in cells),
        "context_metrics": cells,
    }
    for key, reducer in (
        ("min_mem_available_GiB", min),
        ("swap_growth_GiB", max),
        ("major_faults_delta", max),
        ("peak_gtt_used_GiB", max),
        ("peak_vram_used_GiB", max),
        ("peak_temperature_c", max),
        ("peak_power_w", max),
    ):
        present = [cell[key] for cell in cells if key in cell]
        if present:
            result[key] = reducer(present)
    return result


def check_context_regressions(cells: list[dict], baseline: dict | None, min_fraction: float) -> list[str]:
    if baseline is None:
        return []
    base_cells = {int(cell["context_depth"]): cell for cell in baseline.get("context_metrics", [])}
    violations = []
    for cell in cells:
        depth = int(cell["context_depth"])
        base = base_cells.get(depth)
        if base is None:
            violations.append(f"context {depth}: missing baseline cell")
            continue
        for metric in ("decode_tok_s", "prefill_tok_s"):
            floor = min_fraction * float(base[metric])
            value = float(cell[metric])
            if value < floor:
                violations.append(f"context {depth} {metric}: {value:.4g} < {min_fraction}*baseline({base[metric]:.4g})")
    return violations


# ---------------------------------------------------------------------------
# Golden-output correctness
# ---------------------------------------------------------------------------


def load_golden(benchmark_name: str) -> dict | None:
    golden_path = GOLDEN_DIR / f"{benchmark_name}.json"
    if not golden_path.exists():
        return None
    return json.loads(golden_path.read_text())


def build_generation_command(
    llama_cli: Path, model: Path, draft: Path | None, prompt: str, params: dict, n_draft: int,
    runtime_flags: dict[str, str] | None = None,
) -> list[str]:
    """Build the llama-cli invocation used for both correctness generation and golden capture."""
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
    if runtime_flags:
        cmd = apply_runtime_flags(cmd, runtime_flags)
    return cmd


def extract_generated_text(stdout: str) -> str:
    """
    Extract the model generation from llama-cli stdout.

    llama-cli prints the generation followed by a `llama_print_timings:` summary
    block; the prompt echo is already suppressed via --no-display-prompt.
    """
    return stdout.split("llama_print_timings:", 1)[0].strip()


def generate_text(
    llama_cli: Path, model: Path, draft: Path | None, prompt: str, params: dict, n_draft: int,
    runtime_flags: dict[str, str] | None = None,
) -> str:
    cmd = build_generation_command(llama_cli, model, draft, prompt, params, n_draft, runtime_flags)
    result = run(cmd, timeout=300)
    return extract_generated_text(result.stdout)


def check_correctness(
    benchmark_name: str, model: Path, draft: Path | None, n_draft: int, runtime_flags: dict[str, str],
) -> tuple[bool, list[dict]]:
    """
    Compare generated text against golden outputs. Raises if correctness cannot
    be evaluated; returns (all_passed, per_prompt results) otherwise.
    """
    golden = load_golden(benchmark_name)
    if golden is None:
        raise RuntimeError(f"no golden outputs for benchmark '{benchmark_name}'; run scripts/generate_golden.py")

    layout = SourceLayout.resolve()
    layout.require_capability("llama-tools")
    llama_cli = layout.build_dir(detect_backend()) / "bin" / "llama-cli"
    if not llama_cli.exists():
        raise RuntimeError(f"{llama_cli} not found; run prepare.py (or use --simulate)")
    if golden.get("generated_at") == "NOT_YET_GENERATED":
        raise RuntimeError(f"golden outputs for benchmark '{benchmark_name}' are placeholders; run scripts/generate_golden.py")

    params = golden.get("parameters", {})
    results = []
    all_passed = True
    for expected in golden.get("outputs", []):
        actual = generate_text(llama_cli, model, draft, expected["prompt"], params, n_draft, runtime_flags)
        passed = actual == expected["text"]
        all_passed = all_passed and passed
        results.append({
            "prompt": expected["prompt"],
            "passed": passed,
            "expected": expected["text"],
            "actual": actual,
        })
    return all_passed, results


def resolve_correctness(
    benchmark_name: str, spec: dict, model: Path, draft: Path | None, n_draft: int,
    runtime_flags: dict[str, str],
) -> tuple[bool, list[dict]]:
    """
    Golden-output comparison unless the benchmark declares "quality": "kl"
    (shadow benches: user prompts have no goldens; KL against the frozen
    baseline reference is the sole quality oracle, enforced in
    run_single_benchmark via kl_text).
    """
    if spec.get("quality") == "kl":
        return True, []
    return check_correctness(benchmark_name, model, draft, n_draft, runtime_flags)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_score(metrics: dict[str, float], correct: bool) -> float:
    """
    The score is the single maximized axis: decode throughput. Correctness is a
    hard constraint (0.0 on failure); resource/regression bounds are enforced
    separately via objective.check_constraints, which zeroes the score on
    violation exactly like a correctness failure.
    """
    if not correct:
        return 0.0
    return metrics["decode_tok_s"]


def save_baseline_metrics(per_benchmark: list[dict]) -> None:
    """Persist per-benchmark baseline metrics (numeric fields only) for candidate
    runs to load as the reference for relative constraints."""
    data = {
        r["benchmark"]: {
            key: val for key, val in r.items()
            if isinstance(val, (int, float)) or key == "context_metrics"
        }
        for r in per_benchmark
    }
    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_PATH.write_text(json.dumps(data, indent=2))


def load_baseline_metrics() -> dict | None:
    if not BASELINE_METRICS_PATH.exists():
        return None
    return json.loads(BASELINE_METRICS_PATH.read_text())


# ---------------------------------------------------------------------------
# Multi-benchmark aggregation
# ---------------------------------------------------------------------------


def list_benchmarks() -> list[str]:
    return sorted(p.stem for p in BENCHMARKS_DIR.glob("*.json") if p.name != "manifest.json")


def _simulated_metrics() -> dict[str, float]:
    """Deterministic fake measurements for --simulate plumbing tests only."""
    return {
        "decode_tok_s": 120.0,
        "decode_tok_s_stddev": 4.0,
        "prefill_tok_s": 2500.0,
        "prefill_tok_s_stddev": 100.0,
        "acceptance_rate": 1.0,
        "peak_mem_GiB": 18.0,
    }


def run_single_benchmark(
    benchmark_name: str,
    build_time: float,
    runtime_flags: dict[str, str],
    simulate: bool = False,
    profile_dir: str | None = None,
    baseline_metrics: dict | None = None,
    k: float = 1.0,
    enforce_constraints: bool = False,
) -> dict:
    benchmark = json.loads((BENCHMARKS_DIR / f"{benchmark_name}.json").read_text())

    kl_violations: list[str] = []
    if simulate:
        metrics = _simulated_metrics()
        correct = True
        details: list[dict] = []
    else:
        entry = benchmark.get("manifest_entry", benchmark_name)
        model = resolve_model(entry, "target")
        draft = resolve_model(entry, "draft")
        n_draft = benchmark.get("n_draft", 15)
        if model is None or not model.exists():
            raise RuntimeError(f"target model for benchmark '{benchmark_name}' not found; run prepare.py (or use --simulate)")
        profile_path = os.path.join(profile_dir, benchmark_name) if profile_dir else None
        if profile_path:
            Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        contexts = [int(value) for value in benchmark.get("contexts", [])]
        if contexts:
            cells = []
            for depth in contexts:
                cell_profile = f"{profile_path}-ctx{depth}" if profile_path else None
                cell = benchmark_with_llama_bench(
                    model, draft, n_draft, benchmark.get("llama_bench_args", {}), runtime_flags,
                    profile_path=cell_profile, context_depth=depth,
                )
                cell["context_depth"] = depth
                cells.append(cell)
            metrics = aggregate_context_metrics(cells, int(benchmark.get("primary_context_count", 2)))
        else:
            metrics = benchmark_with_llama_bench(
                model, draft, n_draft, benchmark.get("llama_bench_args", {}), runtime_flags,
                profile_path=profile_path,
            )
        correct, details = resolve_correctness(benchmark_name, benchmark, model, draft, n_draft, runtime_flags)
        # Tier-1 quality oracle: KL against the frozen baseline reference.
        # check_kl raises when kl_text is declared but the reference is missing.
        if benchmark.get("kl_text"):
            kl_metrics = check_kl(benchmark_name, model, extra=apply_runtime_flags([], runtime_flags))
            metrics.update(kl_metrics)
            kl_violations = kl_gate(kl_metrics, resolve_kl_tau(benchmark))
            for v in kl_violations:
                print(f"KL VIOLATION [{benchmark_name}]: {v}")

    metrics["build_time_s"] = build_time
    metrics["correctness"] = "pass" if correct else "FAIL"
    violations: list[str] = list(kl_violations)
    if enforce_constraints:
        base = baseline_metrics.get(benchmark_name) if baseline_metrics else None
        violations += check_constraints(metrics, base, benchmark, k)
        if metrics.get("context_metrics"):
            min_fraction = float(benchmark.get("context_min_frac_of_baseline", 0.95))
            violations += check_context_regressions(metrics["context_metrics"], base, min_fraction)
        for v in violations[len(kl_violations):]:
            print(f"CONSTRAINT VIOLATION [{benchmark_name}]: {v}")
    score = 0.0 if violations else compute_score(metrics, correct)

    return {
        "benchmark": benchmark_name,
        "score": score,
        "score_stddev": propagate_score_stddev(metrics) if score else 0.0,
        "constraint_violations": violations,
        **metrics,
        "correctness_details": details,
    }


def aggregate_scores(results: list[dict]) -> dict[str, float]:
    if not results:
        return {"score": 0.0, "score_stddev": 0.0}
    n = len(results)
    combined_sigma = math.sqrt(sum(r.get("score_stddev", 0.0) ** 2 for r in results)) / n
    agg = {
        "score": sum(r["score"] for r in results) / n,
        "score_stddev": combined_sigma,
        "decode_tok_s": sum(r["decode_tok_s"] for r in results) / n,
        "prefill_tok_s": sum(r["prefill_tok_s"] for r in results) / n,
        "peak_mem_GiB": sum(r["peak_mem_GiB"] for r in results) / n,
    }
    rates = [r["acceptance_rate"] for r in results if "acceptance_rate" in r]
    if rates:  # diagnostic only; absent when no benchmark used a draft model
        agg["acceptance_rate"] = sum(rates) / len(rates)
    return agg


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run_harness(baseline: bool = False, simulate: bool = False, profile: bool = False, k: float = 1.0) -> dict:
    profile_dir = str(ROOT / "results" / "profiles") if profile else None
    if simulate:
        exp_info = {"description": "baseline" if baseline else "simulated", "cmake_flags": [], "runtime_flags": {}}
        build_time = 1.0
    else:
        SourceLayout.resolve().require_capability("product-benchmark")
        if baseline:
            reset_lucebox()
            exp_info = {"description": "baseline", "cmake_flags": get_cmake_flags(), "runtime_flags": get_runtime_flags()}
        else:
            reset_lucebox()
            exp_info = apply_experiment()
        build_time = build()

    runtime_flags = exp_info.get("runtime_flags", {})

    benchmark_names = os.environ.get("AUTOGGML_BENCHMARKS", ",".join(list_benchmarks())).split(",")
    benchmark_names = [b.strip() for b in benchmark_names if b.strip()]

    # Constraints gate candidates only: the baseline defines the reference, and
    # --simulate is a plumbing mode with no baseline file to compare against.
    enforce = not baseline and not simulate
    baseline_metrics = load_baseline_metrics() if enforce else None
    per_benchmark = [
        run_single_benchmark(
            name, build_time, runtime_flags, simulate=simulate, profile_dir=profile_dir,
            baseline_metrics=baseline_metrics, k=k, enforce_constraints=enforce,
        )
        for name in benchmark_names
    ]
    if baseline and not simulate:
        save_baseline_metrics(per_benchmark)
    agg = aggregate_scores(per_benchmark)
    any_fail = any(r["correctness"] == "FAIL" for r in per_benchmark)

    summary = {
        "score": agg["score"],
        **agg,
        "build_time_s": build_time,
        "correctness": "pass" if not any_fail else "FAIL",
        "benchmarks": per_benchmark,
        "experiment": exp_info,
    }
    bundle_dir = os.environ.get("AUTOGGML_RESULT_BUNDLE")
    if bundle_dir and not simulate:
        path = Path(bundle_dir)
        path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (path / f"{stamp}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Plumbing test with fake measurements; no real build or models required")
    parser.add_argument("--profile", action="store_true", help="Capture a profiler trace (nsys/rocprof) per benchmark to results/profiles/")
    parser.add_argument("--significance", type=float, default=1.0, help="k in the k*sigma margin for objective constraints")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--target", help="Run through a configured SSH target")
    parser.add_argument("--contract", type=Path, help="Research contract used to namespace remote state")
    parser.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    args = parser.parse_args()

    if args.target and os.environ.get("AUTOGGML_REMOTE_WORKER") != "1":
        from autoggml.contracts import ResearchContract
        from autoggml.remote import SSHWorker
        from autoggml.targets import TargetConfig

        target = TargetConfig.load(args.target)
        contract = ResearchContract.read(args.contract) if args.contract else None
        if contract and args.backend not in contract.backends:
            raise ValueError(f"backend '{args.backend}' is not allowed by the research contract")
        namespace = (
            f"{contract.machine_fingerprint[:16]}-{contract.model_fingerprint[:16]}-{args.backend}"
            if contract else f"uncontracted-{args.backend}"
        )
        backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP"}[args.backend]
        remote_args = []
        if args.baseline:
            remote_args.append("--baseline")
        if args.profile:
            remote_args.append("--profile")
        remote_args += ["--significance", str(args.significance)]
        worker = SSHWorker(target)
        worker.ensure_remote_uv()
        summary = worker.run_harness(remote_args, env={
            backend_var: "ON",
            "AUTOGGML_MODEL_ROOT": target.model_root or f"{target.root}/work/models",
            "AUTOGGML_BUILD_SUBDIR": f"build-{args.backend}",
            "AUTOGGML_STATE_DIR": f"{target.root}/work/state/{namespace}",
            "AUTOGGML_GOLDEN_DIR": f"{target.root}/work/state/{namespace}/golden",
            "AUTOGGML_RESULT_BUNDLE": f"{target.root}/results/runs/{namespace}",
            "AUTOGGML_BENCHMARKS": contract.model if contract else "deepseek-v4-flash",
        })
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.json:
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            summary = run_harness(baseline=args.baseline, simulate=args.simulate, profile=args.profile, k=args.significance)
    else:
        summary = run_harness(baseline=args.baseline, simulate=args.simulate, profile=args.profile, k=args.significance)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    mode = "baseline" if args.baseline else "experiment"
    if args.simulate:
        mode += " (simulated)"
    print("---")
    print(f"mode:              {mode}")
    print(f"score:             {summary['score']:.4f}")
    print(f"score_stddev:      {summary['score_stddev']:.4f}")
    print(f"decode_tok_s:      {summary['decode_tok_s']:.2f}")
    print(f"prefill_tok_s:     {summary['prefill_tok_s']:.2f}")
    if "acceptance_rate" in summary:
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
