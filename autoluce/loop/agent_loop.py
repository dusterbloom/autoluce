"""
Autonomous agent loop for autoluce v2.

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

from autoluce.parallel.concurrency import LockedFrontier
from autoluce.bench.harness import run_harness

from autoluce import ROOT
from autoluce.source_layout import SourceLayout


def _run_remote(args) -> int:
    from autoluce.contracts import ResearchContract
    from autoluce.freeze import contract_namespace
    from autoluce.remote import SSHWorker
    from autoluce.targets import TargetConfig

    SourceLayout.resolve().require_capability("product-benchmark")
    if not args.contract:
        raise ValueError("remote run requires --contract")
    contract = ResearchContract.read(args.contract)
    if args.backend not in contract.backends:
        raise ValueError(f"backend '{args.backend}' is not allowed by the research contract")
    target = TargetConfig.load(args.target)
    namespace = contract_namespace(contract, args.backend)
    root = target.root.rstrip("/")
    state = f"{root}/work/state/{namespace}"
    backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP"}[args.backend]
    remote_args = ["--significance", str(args.significance)]
    if args.baseline:
        remote_args.append("--baseline")
    if args.profile:
        remote_args.append("--profile")
    worker = SSHWorker(target)
    worker.ensure_remote_uv()
    summary = worker.run_harness(remote_args, env={
        backend_var: "ON",
        "AUTOLUCE_MODEL_ROOT": target.model_root or f"{root}/work/models",
        "AUTOLUCE_BUILD_SUBDIR": f"build-{args.backend}",
        "AUTOLUCE_STATE_DIR": state,
        "AUTOLUCE_GOLDEN_DIR": f"{state}/golden",
        "AUTOLUCE_RESULT_BUNDLE": f"{root}/results/runs/{namespace}",
        "AUTOLUCE_BENCHMARKS": contract.model,
        "AUTOLUCE_EXPERIMENT_PATCH": args.experiment_patch or "",
    })

    frontier = LockedFrontier(ROOT / "results" / "frontiers" / namespace, k=args.significance)
    candidate = args.experiment_patch or git_current_commit()
    description = summary.get("experiment", {}).get("description", candidate)
    if summary.get("correctness") != "pass" or summary.get("score", 0.0) <= 0:
        frontier.log_result(candidate, summary, "discard", description)
        print(f"Remote candidate rejected: score={summary.get('score', 0.0):.4f}, correctness={summary.get('correctness')}")
        return 1
    if args.baseline:
        frontier.set_best(candidate, summary, f"baseline: {description}")
        print(f"Remote baseline recorded: {summary['score']:.4f} +/- {summary.get('score_stddev', 0.0):.4f}")
        return 0
    claim = frontier.claim_best_if_significant(candidate, summary, description)
    verdict = "kept" if claim.claimed else "discarded"
    print(f"Remote candidate {verdict}: {summary['score']:.4f} +/- {summary.get('score_stddev', 0.0):.4f}")
    return 0


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


def run_single_experiment(timeout: int = 3600, baseline: bool = False, simulate: bool = False, k: float = 1.0) -> dict:
    start_commit = git_current_commit()
    description = "baseline"

    try:
        summary = run_harness(baseline=baseline, simulate=simulate, k=k)
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
    parser.add_argument("--target", help="Run on a configured SSH target")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    parser.add_argument("--experiment-patch", help="Patch filename under patches/ to apply remotely")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.target:
        sys.exit(_run_remote(args))

    # Frontier root defaults to this checkout so a lone worker just works; override
    # via AUTOLUCE_FRONTIER so workers in separate worktrees funnel into ONE shared
    # leaderboard instead of each writing their own .best_score.json.
    frontier_root = Path(os.environ.get("AUTOLUCE_FRONTIER", str(ROOT)))
    frontier = LockedFrontier(frontier_root, k=args.significance)
    best = frontier.read_best()
    best_score = 0.0 if args.baseline else float(best.get("score", 0.0))
    best_stddev = 0.0 if args.baseline else float(best.get("score_stddev", 0.0))
    start_commit = git_current_commit()

    print(f"Starting experiment at commit {start_commit}")
    print(f"Current best score: {best_score:.4f} ± {best_stddev:.4f}")

    t0 = time.time()
    result = run_single_experiment(timeout=args.timeout, baseline=args.baseline, simulate=args.simulate, k=args.significance)
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
