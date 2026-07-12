"""Score uncertainty and the significance gate for keep/revert."""

from __future__ import annotations

import math


def propagate_score_stddev(metrics: dict[str, float], objective: str = "decode_tok_s") -> float:
    """Return the standard deviation belonging to the selected score metric."""
    return metrics.get(f"{objective}_stddev", 0.0)


def is_significant_improvement(
    new_score: float, new_sigma: float, best_score: float, best_sigma: float, k: float = 1.0
) -> bool:
    """One-sided two-sample test; reduces to strict `new > best` at zero variance."""
    delta = new_score - best_score
    noise = k * math.sqrt(new_sigma ** 2 + best_sigma ** 2)
    return delta > noise
