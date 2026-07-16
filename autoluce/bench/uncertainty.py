"""Score uncertainty and the significance gates for keep/revert.

`is_significant_improvement` is the legacy k*sigma margin gate. Prefer
`is_significant_improvement_samples` (Welch's t-test over per-repetition
samples) whenever both sides carry real sample arrays.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence

from autoluce.bench.statistics import welch_t_test


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


def is_significant_improvement_samples(
    new_samples: Sequence[float], best_samples: Sequence[float], alpha: float = 0.05
) -> bool:
    """Sample-aware gate: one-sided Welch's t-test at the given alpha level.

    Falls back to a strict mean comparison when either side lacks the two
    observations needed for a variance estimate -- that path has no error
    control and exists only so n=1 plumbing runs keep their legacy behavior.
    """
    new = [float(sample) for sample in new_samples]
    best = [float(sample) for sample in best_samples]
    if not new or not best:
        return False
    if len(new) < 2 or len(best) < 2:
        return fmean(new) > fmean(best)
    return welch_t_test(new, best).p_value < alpha
