"""
Clean A/B verification: the gold-standard gate before committing a winner.

Parallel screening (runner.screen) measures candidates against a fixed baseline
that may have moved by the time they finish, so any candidate must be re-checked
here on a single quiet worker: clean-build the optimized code, measure, revert to
baseline, clean-build again, measure, and decide. This kills stale-object risk
(the same risk our preserved build dir introduces) and noise from noisy neighbours.

ab_compare is the pure decision; the build+measure orchestration is the worker's job.
"""

from __future__ import annotations

from autoggml.bench.uncertainty import is_significant_improvement


def ab_compare(baseline_result: dict, optimized_result: dict, k: float = 1.0) -> dict:
    """
    Decide whether `optimized_result` is a real, significant win over `baseline_result`,
    both measured clean-built on the same quiet worker.

    Returns {"verified": bool, "delta": float} on success, or
    {"verified": False, "reason": str} when correctness bars verification.
    """
    if optimized_result.get("correctness") != "pass":
        return {"verified": False, "reason": "optimized correctness failed"}
    if baseline_result.get("correctness") != "pass":
        return {"verified": False, "reason": "baseline correctness failed; cannot verify"}

    delta = optimized_result["score"] - baseline_result["score"]
    verified = is_significant_improvement(
        optimized_result["score"], optimized_result.get("score_stddev", 0.0),
        baseline_result["score"], baseline_result.get("score_stddev", 0.0),
        k,
    )
    return {"verified": verified, "delta": delta}
