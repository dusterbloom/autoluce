"""
Tests for score-uncertainty propagation and the significance gate that
replaces the old `score > best` keep/revert rule.
"""

import math

import pytest

from uncertainty import is_significant_improvement, propagate_score_stddev


def _metrics(**overrides):
    base = {
        "decode_tok_s": 100.0,
        "prefill_tok_s": 2000.0,
        "acceptance_rate": 1.0,
        "peak_mem_GiB": 16.0,
        "build_time_s": 100.0,
    }
    base.update(overrides)
    return base


def test_propagate_score_stddev_relative_error():
    score = (100.0 * 2000.0 * 1.0) / (16.0 * 100.0)  # 125.0
    metrics = _metrics(decode_tok_s_stddev=4.0, prefill_tok_s_stddev=100.0)
    expected = score * math.sqrt((4.0 / 100.0) ** 2 + (100.0 / 2000.0) ** 2)
    assert propagate_score_stddev(metrics, score) == pytest.approx(expected)


def test_propagate_score_stddev_zero_without_uncertainty():
    assert propagate_score_stddev(_metrics(), 125.0) == 0.0


def test_is_significant_improvement_within_noise_is_false():
    # delta=4 < k * sqrt(5^2 + 5^2) ~= 7.07
    assert is_significant_improvement(104.0, 5.0, 100.0, 5.0, k=1.0) is False


def test_is_significant_improvement_beyond_noise_is_true():
    # delta=10 > 7.07
    assert is_significant_improvement(110.0, 5.0, 100.0, 5.0, k=1.0) is True


def test_is_significant_improvement_zero_variance_is_strict_gt():
    assert is_significant_improvement(110.0, 0.0, 100.0, 0.0) is True
    assert is_significant_improvement(100.0, 0.0, 100.0, 0.0) is False
    assert is_significant_improvement(99.0, 0.0, 100.0, 0.0) is False
