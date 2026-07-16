"""Statistical core for research decisions.

Pure functions, no IO, no autoluce imports. These replace the ad-hoc k*sigma
margin rule with real tests:

- Welch's two-sample t-test for unpaired candidate-vs-reference comparisons.
- One-sample t-test on paired block deltas for interleaved (ABBA) designs,
  which cancel slow machine drift instead of modelling it.
- Holm and Benjamini-Hochberg corrections so screening funnels that test many
  candidates do not suffer the winner's curse uncorrected.
- Log-scale uncertainty propagation for geometric-mean score aggregation.

The t-distribution survival function is computed through the regularized
incomplete beta function (continued fraction, Lentz's method), accurate to
~1e-12; no scipy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Sequence

_BETACF_MAX_ITERATIONS = 200
_BETACF_EPSILON = 3e-14
_BETACF_FP_MIN = 1e-300


@dataclass(frozen=True)
class TTestResult:
    """One-sided result for H1 "new mean exceeds reference mean" (or deltas > 0)."""

    statistic: float  # t statistic; +/-inf for deterministic zero-variance comparisons
    df: float         # degrees of freedom (Welch-Satterthwaite); inf when deterministic
    p_value: float    # one-sided p-value
    effect: float     # mean difference (new - reference)


def standard_error(samples: Sequence[float]) -> float:
    """Standard error of the mean; 0.0 when no variance estimate exists (n < 2)."""
    if len(samples) < 2:
        return 0.0
    return stdev(samples) / math.sqrt(len(samples))


def student_t_sf(t: float, df: float) -> float:
    """One-sided survival function P(T > t) for Student's t with df degrees of freedom.

    Via the regularized incomplete beta function: two-sided p = I_x(df/2, 1/2)
    with x = df / (df + t^2).
    """
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    two_sided = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return two_sided / 2.0 if t > 0 else 1.0 - two_sided / 2.0


def welch_t_test(new_samples: Sequence[float], ref_samples: Sequence[float]) -> TTestResult:
    """Welch's unequal-variance t-test, one-sided for H1: mean(new) > mean(ref).

    Raises ValueError when either side has fewer than 2 observations (no variance
    estimate). Zero-variance sides are allowed: if both sides are deterministic the
    comparison reduces to a strict mean comparison with p in {0.0, 1.0}.
    """
    n_new, n_ref = len(new_samples), len(ref_samples)
    if n_new < 2 or n_ref < 2:
        raise ValueError("welch_t_test requires at least 2 samples per side")
    mean_new, mean_ref = fmean(new_samples), fmean(ref_samples)
    var_new, var_ref = _sample_variance(new_samples), _sample_variance(ref_samples)
    effect = mean_new - mean_ref
    se_sq = var_new / n_new + var_ref / n_ref
    if se_sq == 0.0:  # both sides deterministic
        if effect > 0:
            return TTestResult(statistic=math.inf, df=math.inf, p_value=0.0, effect=effect)
        return TTestResult(statistic=-math.inf if effect < 0 else 0.0, df=math.inf, p_value=1.0, effect=effect)
    statistic = effect / math.sqrt(se_sq)
    df = se_sq ** 2 / (
        (var_new / n_new) ** 2 / (n_new - 1) + (var_ref / n_ref) ** 2 / (n_ref - 1)
    )
    return TTestResult(statistic=statistic, df=df, p_value=student_t_sf(statistic, df), effect=effect)


def one_sample_t_test(deltas: Sequence[float]) -> TTestResult:
    """One-sample t-test on paired differences, one-sided for H1: mean > 0.

    Use on abba_block_deltas output: pairing adjacent clean/candidate measurements
    cancels slow drift (thermal, clock, neighbour load) that unpaired tests absorb
    as variance.
    """
    n = len(deltas)
    if n < 2:
        raise ValueError("one_sample_t_test requires at least 2 deltas")
    mean = fmean(deltas)
    var = _sample_variance(deltas)
    if var == 0.0:
        if mean > 0:
            return TTestResult(statistic=math.inf, df=math.inf, p_value=0.0, effect=mean)
        return TTestResult(statistic=-math.inf if mean < 0 else 0.0, df=math.inf, p_value=1.0, effect=mean)
    statistic = mean / (math.sqrt(var) / math.sqrt(n))
    return TTestResult(statistic=statistic, df=float(n - 1), p_value=student_t_sf(statistic, n - 1), effect=mean)


def abba_block_deltas(
    sequence: Sequence[tuple[str, float]],
    clean_label: str = "clean",
    candidate_label: str = "candidate",
) -> list[float]:
    """Pair consecutive measurements of an interleaved sequence into per-block deltas.

    Expects the measurement order the harness emits for interleaved A/B designs
    (A B B A A B ...): every consecutive pair must contain exactly one clean and
    one candidate measurement, in either order. Returns candidate - clean per pair.
    Raises ValueError on malformed sequences -- silently mis-pairing would be worse.
    """
    if len(sequence) % 2 != 0:
        raise ValueError("interleaved sequence must have an even number of measurements")
    deltas = []
    for index in range(0, len(sequence), 2):
        pair = sequence[index:index + 2]
        labels = {label for label, _ in pair}
        if labels != {clean_label, candidate_label}:
            raise ValueError(f"block {index // 2} must pair one {clean_label!r} with one {candidate_label!r}")
        values = dict(pair)
        deltas.append(values[candidate_label] - values[clean_label])
    return deltas


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the original order.

    Controls the family-wise error rate; use when confirming a small set of
    candidates that must each be defensible.
    """
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    envelope = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        envelope = max(envelope, min(1.0, (len(ordered) - rank) * p_value))
        adjusted[original_index] = envelope
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values, in the original order.

    Controls the false discovery rate; the right correction for a screening
    funnel that promotes a shortlist into confirmatory runs.
    """
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    envelope = 1.0
    for rank in range(len(ordered) - 1, -1, -1):
        original_index, p_value = ordered[rank]
        envelope = min(envelope, len(ordered) / (rank + 1) * p_value)
        adjusted[original_index] = min(1.0, envelope)
    return adjusted


def geometric_mean_with_stddev(values: Sequence[float], stddevs: Sequence[float]) -> tuple[float, float]:
    """Geometric mean with first-order (log-scale) uncertainty propagation.

    For independent cells, Var(ln X_i) ~= (sigma_i / x_i)^2, so the sigma of the
    geometric mean is gm * sqrt(sum((sigma_i/x_i)^2)) / n. This replaces the old
    arithmetic-mean formula that was applied to a geometric-mean score.
    Returns (0.0, 0.0) when any value is non-positive, matching the caller's
    existing zero-on-missing semantics.
    """
    n = len(values)
    if n == 0 or any(value <= 0 for value in values):
        return 0.0, 0.0
    gm = math.exp(sum(math.log(value) for value in values) / n)
    relative_variance = sum(
        (sigma / value) ** 2 for value, sigma in zip(values, stddevs) if sigma > 0
    )
    return gm, gm * math.sqrt(relative_variance) / n


def _sample_variance(samples: Sequence[float]) -> float:
    deviation = stdev(samples)
    return deviation * deviation


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) via continued fraction (Numerical Recipes, betai)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETACF_FP_MIN:
        d = _BETACF_FP_MIN
    d = 1.0 / d
    h = d
    for m in range(1, _BETACF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_FP_MIN:
            d = _BETACF_FP_MIN
        c = 1.0 + aa / c
        if abs(c) < _BETACF_FP_MIN:
            c = _BETACF_FP_MIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_FP_MIN:
            d = _BETACF_FP_MIN
        c = 1.0 + aa / c
        if abs(c) < _BETACF_FP_MIN:
            c = _BETACF_FP_MIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPSILON:
            break
    return h
