"""
Tests for interleaved (ABBA) A/B measurement -- the drift-robust frontier gate.

The pure scheduling and analysis logic is verified with scripted arms; no server
or GPU is involved. GPU-level validation (A/A null, known-effect) is run manually
via `autoluce ab`.
"""

import pytest

from autoluce.bench.interleave import (
    CANDIDATE,
    CLEAN,
    abba_schedule,
    analyze_sequence,
    run_interleaved,
)


# --- schedule -------------------------------------------------------------------


def test_abba_schedule_mirrors_pairs_within_block():
    assert abba_schedule(1) == [CLEAN, CANDIDATE, CANDIDATE, CLEAN]
    assert abba_schedule(2) == [CLEAN, CANDIDATE, CANDIDATE, CLEAN] * 2


def test_abba_schedule_requires_at_least_one_block():
    with pytest.raises(ValueError):
        abba_schedule(0)


# --- analysis -------------------------------------------------------------------


def _sequence(clean_values, candidate_values):
    """Zip per-block clean/candidate means into the ABBA activation order."""
    sequence = []
    for c1, c2, b1, b2 in zip(
        clean_values[::2], clean_values[1::2], candidate_values[::2], candidate_values[1::2]
    ):
        sequence += [(CLEAN, c1), (CANDIDATE, b1), (CANDIDATE, b2), (CLEAN, c2)]
    return sequence


def test_analyze_clear_win_is_significant_with_effect():
    # clean ~100, candidate ~110 with tight spread: every paired delta ~ +10.
    sequence = _sequence([100.0, 101.0, 99.0, 100.0], [110.0, 111.0, 109.0, 110.0])
    result = analyze_sequence(sequence, metric="decode_tok_s")
    assert result.significant is True
    assert result.effect == pytest.approx(10.0, abs=1.0)
    assert result.effect_pct == pytest.approx(10.0, abs=1.0)
    assert result.clean_mean == pytest.approx(100.0)
    assert result.candidate_mean == pytest.approx(110.0)
    assert result.pairs == 4  # 2 ABBA blocks x 2 consecutive pairs each
    assert result.test.p_value < 0.05


def test_analyze_null_stays_not_significant():
    # Same distribution both arms; deltas centered on zero.
    sequence = _sequence([100.0, 101.0, 99.0, 100.5], [100.2, 99.8, 100.4, 99.9])
    result = analyze_sequence(sequence)
    assert result.significant is False
    assert result.regression is False
    assert result.test.p_value > 0.05


def test_analyze_clear_regression_flags_regression_not_improvement():
    # Candidate ~90 vs clean ~100: the improvement gate must stay false while the
    # regression side fires -- a frontier tool may never call harm "significant".
    sequence = _sequence([100.0, 101.0, 99.0, 100.0], [90.0, 91.0, 89.0, 90.0])
    result = analyze_sequence(sequence)
    assert result.significant is False
    assert result.regression is True
    assert result.effect == pytest.approx(-10.0, abs=1.0)


def test_analyze_rejects_malformed_sequence():
    with pytest.raises(ValueError):
        analyze_sequence([(CLEAN, 100.0), (CLEAN, 101.0)])


# --- orchestration with a scripted arm ------------------------------------------


def test_run_interleaved_follows_schedule_and_analyzes():
    scripted = {CLEAN: iter([100.0, 101.0, 99.0, 100.0]), CANDIDATE: iter([110.0, 111.0, 109.0, 110.0])}
    calls = []

    def measure(label: str) -> float:
        calls.append(label)
        return next(scripted[label])

    result = run_interleaved(measure, blocks=2)
    assert calls == [CLEAN, CANDIDATE, CANDIDATE, CLEAN] * 2
    assert result.significant is True
    assert result.effect == pytest.approx(10.0, abs=1.0)


def test_run_interleaved_reports_progress():
    progress = []

    def measure(label: str) -> float:
        return 100.0 if label == CLEAN else 101.0

    run_interleaved(measure, blocks=1, on_progress=lambda *args: progress.append(args))
    assert len(progress) == 4
    assert progress[-1][0] == 4 and progress[-1][1] == 4  # (step, total)


def test_run_interleaved_rejects_zero_blocks():
    with pytest.raises(ValueError):
        run_interleaved(lambda label: 1.0, blocks=0)
