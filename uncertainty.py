"""
Score-uncertainty propagation and the significance gate for keep/revert.

The score is a multiplicative combination of measured quantities, so a measured
stddev on any component contributes to the score's stddev via relative-error
propagation. Components without a measured stddev contribute nothing, which
keeps the gate non-regressive: no variance data => strict `new > best`.
"""

from __future__ import annotations

import math

SCORE_COMPONENTS = ("decode_tok_s", "prefill_tok_s", "acceptance_rate", "peak_mem_GiB")


def propagate_score_stddev(metrics: dict[str, float], score: float) -> float:
    """Relative-error propagation of the multiplicative score's stddev."""
    rel_var = 0.0
    for key in SCORE_COMPONENTS:
        mean = metrics.get(key, 0.0)
        sigma = metrics.get(f"{key}_stddev", 0.0)
        if mean:
            rel_var += (sigma / mean) ** 2
    return score * math.sqrt(rel_var)


def is_significant_improvement(
    new_score: float, new_sigma: float, best_score: float, best_sigma: float, k: float = 1.0
) -> bool:
    """One-sided two-sample test; reduces to strict `new > best` at zero variance."""
    delta = new_score - best_score
    noise = k * math.sqrt(new_sigma ** 2 + best_sigma ** 2)
    return delta > noise
