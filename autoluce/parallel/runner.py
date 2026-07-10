"""
Parallel experiment fan-out: dispatch specs concurrently and screen for winners.

The run_fn is injected (dependency inversion): the dispatcher is agnostic to
whether a spec runs as a local subprocess, over SSH to another host, or via
`sky exec` on a cloud VM. Worker adapters are thin and added where compute lives.

Parallel execution is a SCREENING phase: candidates found here must still pass
clean A/B verification (verify.ab_compare) before committing, because they were
measured against a fixed baseline while the best may have moved.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from autoluce.parallel.concurrency import ClaimResult, LockedFrontier
from autoluce.bench.uncertainty import is_significant_improvement
from autoluce.parallel.worktree import ensure_worktree


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    description: str
    target: str = "rtx3090_cuda"
    patch: str | None = None
    runtime_flags: dict[str, str] = field(default_factory=dict)


# A run_fn materializes the spec, builds, benchmarks, and returns at least
# {score, score_stddev, correctness}. Injected so dispatch is testable offline.
RunFn = Callable[[ExperimentSpec], dict]


def dispatch(specs: list[ExperimentSpec], run_fn: RunFn, max_parallel: int) -> list[tuple[ExperimentSpec, dict]]:
    """Run specs concurrently (workers are I/O-bound on subprocess/SSH) and return
    (spec, result) pairs in completion order. Exceptions become crash results."""
    results: list[tuple[ExperimentSpec, dict]] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(run_fn, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                results.append((spec, fut.result()))
            except Exception as e:
                results.append((spec, {"score": 0.0, "score_stddev": 0.0, "correctness": "FAIL", "crash": str(e)}))
    return results


def screen(
    specs: list[ExperimentSpec],
    run_fn: RunFn,
    baseline_score: float,
    baseline_sigma: float,
    max_parallel: int,
    k: float = 1.0,
) -> list[tuple[ExperimentSpec, dict]]:
    """Parallel screen: return specs whose result is a significant improvement over
    the baseline, sorted by score descending. Correctness failures and crashes drop."""
    winners: list[tuple[ExperimentSpec, dict]] = []
    for spec, result in dispatch(specs, run_fn, max_parallel):
        if result.get("correctness") != "pass":
            continue
        if is_significant_improvement(
            result.get("score", 0.0), result.get("score_stddev", 0.0), baseline_score, baseline_sigma, k
        ):
            winners.append((spec, result))
    winners.sort(key=lambda sr: sr[1]["score"], reverse=True)
    return winners


def run_parallel(
    specs: list[ExperimentSpec],
    run_fn: RunFn,
    frontier: LockedFrontier,
    max_parallel: int,
) -> list[tuple[ExperimentSpec, ClaimResult]]:
    """Run specs concurrently; funnel each passing result into the LIVE frontier.

    Unlike screen (which tests against a fixed snapshot and returns winners), run_parallel
    claims each result against the frontier as it stands now: the frontier advances as
    better results land, and a later-finished spec is judged against the moved frontier
    -- the parallel-safe funnel. Correctness failures are skipped (never claimed).
    Returns (spec, ClaimResult) in completion order.
    """
    claimed: list[tuple[ExperimentSpec, ClaimResult]] = []
    for spec, result in dispatch(specs, run_fn, max_parallel):
        if result.get("correctness") != "pass":
            continue
        claim = frontier.claim_best_if_significant(
            commit=result.get("commit", spec.id),
            summary={
                "score": result.get("score", 0.0),
                "score_stddev": result.get("score_stddev", 0.0),
                "decode_tok_s": result.get("decode_tok_s", 0.0),
                "prefill_tok_s": result.get("prefill_tok_s", 0.0),
                "acceptance_rate": result.get("acceptance_rate", 0.0),
                "peak_mem_GiB": result.get("peak_mem_GiB", 0.0),
            },
            description=spec.description,
        )
        claimed.append((spec, claim))
    return claimed


# --- local run_fn: gives run_parallel + worktree a real in-tree consumer -----------

def _harness_command(worktree: Path) -> list[str]:
    """Measure command run inside a worktree (score only -- the claim is run_parallel's)."""
    return ["uv", "run", "python", "-m", "autoluce.bench.harness", "--json"]


def _current_commit(worktree: Path) -> str:
    """Short HEAD of the worktree, or '' if it can't be resolved."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=worktree,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _parse_harness_json(stdout: str, commit: str, spec_id: str) -> dict:
    """Parse `harness --json` output into a run_fn result dict. Malformed -> crash dict."""
    try:
        summary = json.loads(stdout)
    except Exception as e:
        return {"score": 0.0, "score_stddev": 0.0, "correctness": "FAIL",
                "commit": commit or spec_id, "crash": f"harness json parse: {e}"}
    return {
        "score": float(summary.get("score", 0.0)),
        "score_stddev": float(summary.get("score_stddev", 0.0)),
        "correctness": summary.get("correctness", "FAIL"),
        "commit": commit or spec_id,
    }


def local_run(spec: ExperimentSpec, repo_root: Path, run_subprocess=subprocess.run) -> dict:
    """Run one spec in its own worktree and return the measured result dict.

    Ensures an isolated worktree (worktree module's first in-tree consumer), invokes
    `autoluce.bench.harness --json` there via the injected transport, and parses the score. The claim
    is deliberately NOT made here -- run_parallel is the funnel. `commit` in the result is
    the worktree's real HEAD, so a claimed result points at a valid git ref.

    This measures whatever `experiment.py` the worktree already contains. For true parallel
    SEARCH (a different experiment per worktree), materialize `spec.patch` / `runtime_flags`
    into the worktree's experiment.py first -- that step is intentionally not baked in (it
    is heavy and domain-specific). This is the local run_fn giving run_parallel + worktree a
    real consumer; wire it as `run_parallel(specs, lambda s: local_run(s, repo_root), ...)`.
    """
    wt = ensure_worktree(Path(repo_root), spec.id)
    commit = _current_commit(wt)
    proc = run_subprocess(_harness_command(wt), cwd=wt, capture_output=True, text=True)
    return _parse_harness_json(proc.stdout, commit, spec.id)
