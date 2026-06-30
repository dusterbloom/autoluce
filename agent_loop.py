"""
Autonomous agent loop for autoggml v2.

Run one experiment, compare the score to the current best, and either
keep the commit or revert it. Logs results to results.tsv.

Usage:
    uv run agent_loop.py

The loop is meant to be driven by an AI agent editing experiment.py,
committing, and then calling this script. It can also be run manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from harness import run_harness
from uncertainty import is_significant_improvement

ROOT = Path(__file__).resolve().parent
RESULTS_TSV = ROOT / "results.tsv"
RUN_LOG = ROOT / "run.log"
BEST_SCORE_FILE = ROOT / ".best_score.json"


def git_current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_reset(commit: str) -> None:
    subprocess.run(["git", "reset", "--hard", commit], cwd=ROOT, check=True, text=True)


def load_best() -> dict:
    if BEST_SCORE_FILE.exists():
        return json.loads(BEST_SCORE_FILE.read_text())
    return {"score": 0.0, "score_stddev": 0.0, "commit": "", "updated_at": ""}


def load_best_score() -> float:
    return float(load_best().get("score", 0.0))


def load_best_stddev() -> float:
    return float(load_best().get("score_stddev", 0.0))


def load_best_commit() -> str | None:
    commit = load_best().get("commit", "")
    return commit if commit else None


def save_best_score(score: float, score_stddev: float, commit: str) -> None:
    BEST_SCORE_FILE.write_text(json.dumps({
        "score": score,
        "score_stddev": score_stddev,
        "commit": commit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def default_summary() -> dict:
    """Fallback summary for crashes/timeouts."""
    return {
        "score": 0.0,
        "score_stddev": 0.0,
        "decode_tok_s": 0.0,
        "prefill_tok_s": 0.0,
        "acceptance_rate": 0.0,
        "peak_mem_GiB": 0.0,
    }


def log_result(commit: str, summary: dict, status: str, description: str) -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tscore\tscore_stddev\tdecode_tok_s\tprefill_tok_s\tacceptance_rate\tpeak_mem_GiB\tstatus\tdescription\n")

    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow([
            commit,
            f"{summary['score']:.6f}",
            f"{summary.get('score_stddev', 0.0):.6f}",
            f"{summary.get('decode_tok_s', 0.0):.2f}",
            f"{summary.get('prefill_tok_s', 0.0):.2f}",
            f"{summary.get('acceptance_rate', 0.0):.4f}",
            f"{summary.get('peak_mem_GiB', 0.0):.2f}",
            status,
            description,
        ])


def run_single_experiment(timeout: int = 3600, baseline: bool = False, simulate: bool = False) -> dict:
    start_commit = git_current_commit()
    description = "baseline"

    try:
        summary = run_harness(baseline=baseline, simulate=simulate)
        description = summary.get("experiment", {}).get("description", "unknown")
    except subprocess.TimeoutExpired:
        return {
            "commit": start_commit,
            "score": 0.0,
            "status": "crash",
            "description": "timeout",
            "summary": default_summary(),
        }
    except Exception as e:
        return {
            "commit": start_commit,
            "score": 0.0,
            "status": "crash",
            "description": f"exception: {e}",
            "summary": default_summary(),
        }

    return {
        "commit": start_commit,
        "score": summary.get("score", 0.0),
        "status": "keep" if summary.get("correctness") == "pass" else "discard",
        "description": description,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run baseline without keeping/reverting")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-experiment timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify git state")
    parser.add_argument("--simulate", action="store_true", help="Plumbing test; no git ops or best-score writes")
    parser.add_argument("--significance", type=float, default=1.0, help="Keep only if improvement exceeds k*combined_sigma")
    args = parser.parse_args()

    best_score = 0.0 if args.baseline else load_best_score()
    best_stddev = 0.0 if args.baseline else load_best_stddev()
    start_commit = git_current_commit()

    print(f"Starting experiment at commit {start_commit}")
    print(f"Current best score: {best_score:.4f} ± {best_stddev:.4f}")

    t0 = time.time()
    result = run_single_experiment(timeout=args.timeout, baseline=args.baseline, simulate=args.simulate)
    elapsed = time.time() - t0

    commit = result["commit"]
    score = result["score"]
    status = result["status"]
    description = result["description"]
    summary = result["summary"]
    score_stddev = summary.get("score_stddev", 0.0)

    print(f"Score: {score:.4f} ± {score_stddev:.4f} (best: {best_score:.4f} ± {best_stddev:.4f})")
    print(f"Elapsed: {elapsed:.1f}s")

    if args.simulate:
        print("Simulation complete; git state and best score unchanged.")
        return

    if status == "crash":
        label = "Baseline" if args.baseline else "Experiment"
        print(f"{label} crashed: {description}")
        if args.baseline:
            sys.exit(1)
        log_result(commit, summary, "crash", description)
        if not args.dry_run:
            best_commit = load_best_commit()
            if best_commit:
                print(f"Reverting to best commit {best_commit}...")
                git_reset(best_commit)
            else:
                print("No recorded best commit; leaving worktree as-is.")
        return

    if args.baseline:
        log_result(commit, summary, "keep", f"baseline: {description}")
        save_best_score(score, score_stddev, commit)
        print("Baseline recorded.")
        return

    if is_significant_improvement(score, score_stddev, best_score, best_stddev, k=args.significance):
        print(f"Significant improvement (>{args.significance}σ); keeping commit.")
        log_result(commit, summary, "keep", description)
        if not args.dry_run:
            save_best_score(score, score_stddev, commit)
    else:
        print("Improvement not significant; reverting.")
        log_result(commit, summary, "discard", description)
        if not args.dry_run:
            best_commit = load_best_commit()
            if best_commit:
                print(f"Reverting to best commit {best_commit}...")
                git_reset(best_commit)
            else:
                print("No recorded best commit; leaving worktree as-is.")


if __name__ == "__main__":
    main()
