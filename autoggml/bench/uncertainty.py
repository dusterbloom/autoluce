"""
Score-uncertainty and the significance gate for keep/revert.

The score is decode_tok_s directly (resource/regression bounds are enforced
separately by objective.check_constraints), so the score's stddev is just the
measured decode stddev. No variance data => 0.0, which keeps the gate
non-regressive: strict `new > best`.
"""

from __future__ import annotations

import math


def propagate_score_stddev(metrics: dict[str, float]) -> float:
    """Score == decode_tok_s, so its sigma is the measured decode stddev."""
    return metrics.get("decode_tok_s_stddev", 0.0)


def is_significant_improvement(
    new_score: float, new_sigma: float, best_score: float, best_sigma: float, k: float = 1.0
) -> bool:
    """One-sided two-sample test; reduces to strict `new > best` at zero variance."""
    delta = new_score - best_score
    noise = k * math.sqrt(new_sigma ** 2 + best_sigma ** 2)
    return delta > noise
