"""
Tests for the clean A/B verification decision: after parallel screening finds a
candidate, re-measure optimized and baseline clean-built on the same quiet worker
and decide whether the win is real. The decision is pure; the live build+measure
is the worker's job.
"""

from autoggml.loop.verify import ab_compare


def _result(score, sigma, correctness="pass"):
    return {"score": score, "score_stddev": sigma, "correctness": correctness}


def test_ab_compare_significant_win_verifies_true():
    out = ab_compare(_result(100.0, 1.0), _result(120.0, 1.0), k=1.0)
    assert out["verified"] is True
    assert out["delta"] == 20.0


def test_ab_compare_within_noise_does_not_verify():
    # delta 3 vs noise sqrt(2^2 + 2^2) ~= 2.83 -> barely significant at k=1.0,
    # so use a tighter margin to make it clearly within noise.
    out = ab_compare(_result(100.0, 2.0), _result(101.0, 2.0), k=1.0)
    assert out["verified"] is False


def test_ab_compare_optimized_correctness_fail_rejects():
    out = ab_compare(_result(100.0, 1.0), _result(999.0, 0.0, correctness="FAIL"), k=1.0)
    assert out["verified"] is False
    assert "correctness" in out["reason"]


def test_ab_compare_baseline_correctness_fail_is_inconclusive():
    out = ab_compare(_result(100.0, 1.0, correctness="FAIL"), _result(120.0, 1.0), k=1.0)
    assert out["verified"] is False
    assert "baseline" in out["reason"]
