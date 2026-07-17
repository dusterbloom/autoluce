"""Interleaved (ABBA) A/B measurement: the drift-robust comparison primitive.

GPU clocks, thermals, and neighbour load move an apparent result by several
percent, so comparing a candidate against a historical or even sequential
baseline conflates the treatment with machine state. Interleaving the arms in
mirrored ABBA blocks and testing the paired deltas cancels slow drift instead
of modelling it: every block contains both arms measured seconds apart, and the
mirrored pattern exposes each arm equally to first/second position in a pair.

This module is pure orchestration: the caller injects how one activation is
measured (launch server, run repetitions, tear down, return the block value).
Analysis reuses the paired one-sample t-test from `autoluce.bench.statistics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Callable, Sequence

from autoluce.bench.statistics import TTestResult, abba_block_deltas, one_sample_t_test

CLEAN = "clean"
CANDIDATE = "candidate"


def abba_schedule(blocks: int) -> list[str]:
    """The activation order: clean, candidate, candidate, clean, repeated per block."""
    if blocks < 1:
        raise ValueError("interleaved measurement requires at least one ABBA block")
    return [label for _ in range(blocks) for label in (CLEAN, CANDIDATE, CANDIDATE, CLEAN)]


@dataclass(frozen=True)
class InterleavedResult:
    """Paired comparison over ABBA block deltas."""

    metric: str
    pairs: int             # number of paired deltas -- the sample size the t-test uses
    clean_mean: float
    candidate_mean: float
    effect: float          # mean paired delta (candidate - clean), in metric units
    effect_pct: float      # effect relative to the clean mean, percent (nan if undefined)
    deltas: tuple[float, ...]
    test: TTestResult
    significant: bool      # one-sided paired t-test for IMPROVEMENT at the requested alpha
    regression: bool       # one-sided paired t-test for REGRESSION (1 - p) at the same alpha
    sequence: tuple[tuple[str, float], ...] = field(repr=False)


def analyze_sequence(
    sequence: Sequence[tuple[str, float]], metric: str = "decode_tok_s", alpha: float = 0.05
) -> InterleavedResult:
    """Analyze a measured interleaved sequence. Raises ValueError if malformed."""
    deltas = abba_block_deltas(sequence, CLEAN, CANDIDATE)
    test = one_sample_t_test(deltas)
    clean_values = [value for label, value in sequence if label == CLEAN]
    candidate_values = [value for label, value in sequence if label == CANDIDATE]
    clean_mean = fmean(clean_values)
    candidate_mean = fmean(candidate_values)
    effect_pct = (test.effect / clean_mean * 100.0) if clean_mean > 0 else float("nan")
    return InterleavedResult(
        metric=metric,
        pairs=len(deltas),
        clean_mean=clean_mean,
        candidate_mean=candidate_mean,
        effect=test.effect,
        effect_pct=effect_pct,
        deltas=tuple(deltas),
        test=test,
        significant=test.p_value < alpha,
        # The t distribution is symmetric, so the one-sided regression test is
        # the complement of the improvement test.
        regression=(1.0 - test.p_value) < alpha,
        sequence=tuple(sequence),
    )


def run_interleaved(
    measure: Callable[[str], float],
    blocks: int = 8,
    metric: str = "decode_tok_s",
    alpha: float = 0.05,
    on_progress: Callable[[int, int, str, float], None] | None = None,
) -> InterleavedResult:
    """Run an ABBA-interleaved comparison, calling `measure(label)` per activation.

    `measure` receives CLEAN or CANDIDATE and must activate that arm in a fresh
    state (e.g. a newly launched server), measure it, tear it down, and return
    the block value: the mean of the within-activation repetitions of the
    objective metric. Every activation is a fresh process so within-session
    carryover (cache state, allocator fragmentation) cannot leak between blocks.
    """
    schedule = abba_schedule(blocks)
    sequence: list[tuple[str, float]] = []
    for index, label in enumerate(schedule):
        value = float(measure(label))
        sequence.append((label, value))
        if on_progress is not None:
            on_progress(index + 1, len(schedule), label, value)
    return analyze_sequence(sequence, metric=metric, alpha=alpha)
