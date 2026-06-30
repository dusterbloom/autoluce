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

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from uncertainty import is_significant_improvement


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
