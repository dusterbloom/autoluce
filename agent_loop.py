"""
Autonomous agent loop for autoggml v2.

Run one experiment, compare the score to the current best, and either keep the
commit or revert it. Logs results to results.tsv.

All shared-state access (.best_score.json + results.tsv) goes through LockedFrontier,
so concurrent invocations -- multiple workers in git worktrees, or multiple hosts
funneling into one shared checkout -- are safe: the keep/discard decision re-verifies
significance against the LIVE frontier under a file lock, not the snapshot read at the
start of the run.

Usage:
    uv run agent_loop.py

The loop is meant to be driven by an AI agent editing experiment.py, committing, and
then calling this script. It can also be run manually.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from concurrency import LockedFrontier
from harness import run_harness

ROOT = Path(__file__).resolve().parent


def git_current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_reset(commit: str) -> None:
    subprocess.run(["git", "reset", "--hard", commit], cwd=ROOT, check=True, text=True)


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

    # Frontier root defaults to this checkout so a lone worker just works; override
    # via AUTOGGML_FRONTIER so workers in separate worktrees funnel into ONE shared
    # leaderboard instead of each writing their own .best_score.json.
    frontier_root = Path(os.environ.get("AUTOGGML_FRONTIER", str(ROOT)))
    frontier = LockedFrontier(frontier_root, k=args.significance)
    best = frontier.read_best()
    best_score = 0.0 if args.baseline else float(best.get("score", 0.0))
    best_stddev = 0.0 if args.baseline else float(best.get("score_stddev", 0.0))
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
        frontier.log_result(commit, summary, "crash", description)
        if not args.dry_run:
            best_commit = frontier.read_best().get("commit", "")
            if best_commit:
                print(f"Reverting to best commit {best_commit}...")
                git_reset(best_commit)
            else:
                print("No recorded best commit; leaving worktree as-is.")
        return

    if args.baseline:
        frontier.set_best(commit, summary, f"baseline: {description}")
        print("Baseline recorded.")
        return

    claim = frontier.claim_best_if_significant(commit, summary, description)
    if claim.claimed:
        print(f"Significant improvement (>{args.significance}σ); keeping commit.")
    else:
        print("Improvement not significant; reverting.")
        if not args.dry_run:
            if claim.best_commit:
                print(f"Reverting to best commit {claim.best_commit}...")
                git_reset(claim.best_commit)
            else:
                print("No recorded best commit; leaving worktree as-is.")


if __name__ == "__main__":
    main()
