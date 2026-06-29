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
import time
from datetime import datetime, timezone
from pathlib import Path

from harness import run_harness

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


def load_best_score() -> float:
    if BEST_SCORE_FILE.exists():
        data = json.loads(BEST_SCORE_FILE.read_text())
        return float(data.get("score", 0.0))
    return 0.0


def save_best_score(score: float, commit: str) -> None:
    BEST_SCORE_FILE.write_text(json.dumps({"score": score, "commit": commit, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2))


def log_result(commit: str, summary: dict, status: str, description: str) -> None:
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tscore\tdecode_tok_s\tprefill_tok_s\tacceptance_rate\tpeak_mem_GiB\tstatus\tdescription\n")

    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow([
            commit,
            f"{summary['score']:.6f}",
            f"{summary.get('decode_tok_s', 0.0):.2f}",
            f"{summary.get('prefill_tok_s', 0.0):.2f}",
            f"{summary.get('acceptance_rate', 0.0):.4f}",
            f"{summary.get('peak_mem_GiB', 0.0):.2f}",
            status,
            description,
        ])


def run_single_experiment(timeout: int = 3600, baseline: bool = False) -> dict:
    start_commit = git_current_commit()
    description = "baseline"

    try:
        summary = run_harness(baseline=baseline)
        description = summary.get("experiment", {}).get("description", "unknown")
    except subprocess.TimeoutExpired:
        return {
            "commit": start_commit,
            "score": 0.0,
            "status": "crash",
            "description": "timeout",
            "summary": {},
        }
    except Exception as e:
        return {
            "commit": start_commit,
            "score": 0.0,
            "status": "crash",
            "description": f"exception: {e}",
            "summary": {},
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
    args = parser.parse_args()

    best_score = 0.0 if args.baseline else load_best_score()
    start_commit = git_current_commit()

    print(f"Starting experiment at commit {start_commit}")
    print(f"Current best score: {best_score:.4f}")

    t0 = time.time()
    result = run_single_experiment(timeout=args.timeout, baseline=args.baseline)
    elapsed = time.time() - t0

    commit = result["commit"]
    score = result["score"]
    status = result["status"]
    description = result["description"]
    summary = result["summary"]

    print(f"Score: {score:.4f} (best: {best_score:.4f})")
    print(f"Elapsed: {elapsed:.1f}s")

    if args.baseline:
        log_result(commit, summary, "keep", f"baseline: {description}")
        save_best_score(score, commit)
        print("Baseline recorded.")
        return

    if status == "crash":
        print(f"Experiment crashed: {description}")
        log_result(commit, summary, "crash", description)
        if not args.dry_run:
            print("Reverting to best commit...")
            best = json.loads(BEST_SCORE_FILE.read_text())
            git_reset(best["commit"])
        return

    if score > best_score:
        print("Score improved. Keeping commit.")
        log_result(commit, summary, "keep", description)
        if not args.dry_run:
            save_best_score(score, commit)
    else:
        print("Score did not improve. Reverting.")
        log_result(commit, summary, "discard", description)
        if not args.dry_run:
            git_reset(start_commit)


if __name__ == "__main__":
    main()
