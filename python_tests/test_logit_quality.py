from __future__ import annotations

import math

import pytest

from autoluce.bench.logit_quality import (
    aggregate_logit_quality,
    aggregate_logit_quality_gate,
    compare_logits,
    logit_quality_gate,
)


def test_identical_logits_have_zero_error_and_full_topk_overlap():
    metrics = compare_logits([3.0, 2.0, 1.0], [3.0, 2.0, 1.0], top_k=2)

    assert metrics.kl_divergence == pytest.approx(0.0)
    assert metrics.max_abs_error == pytest.approx(0.0)
    assert metrics.top_k_overlap == pytest.approx(1.0)
    assert metrics.reference_argmax == metrics.candidate_argmax == 0
    assert logit_quality_gate(metrics) == []


def test_kl_is_invariant_to_a_uniform_logit_shift():
    metrics = compare_logits([3.0, 2.0, 1.0], [103.0, 102.0, 101.0], top_k=3)

    assert metrics.kl_divergence == pytest.approx(0.0, abs=1e-12)
    assert metrics.max_abs_error == pytest.approx(100.0)
    assert logit_quality_gate(metrics) == []


def test_near_tied_argmax_swap_is_measured_without_implying_corruption():
    metrics = compare_logits([1.0001, 1.0, -2.0], [1.0, 1.0001, -2.0], top_k=2)

    assert metrics.reference_argmax == 0
    assert metrics.candidate_argmax == 1
    assert metrics.argmax_changed is True
    assert metrics.top_k_overlap == pytest.approx(1.0)
    assert metrics.kl_divergence < 1e-6
    assert logit_quality_gate(metrics, kl_tau=0.01, min_top_k_overlap=1.0) == []


@pytest.mark.parametrize(
    "reference,candidate,message",
    [
        ([], [], "non-empty"),
        ([1.0], [1.0, 2.0], "same shape"),
        ([math.nan], [1.0], "finite"),
        ([1.0], [math.inf], "finite"),
    ],
)
def test_logit_comparison_fails_closed(reference, candidate, message):
    with pytest.raises(ValueError, match=message):
        compare_logits(reference, candidate)


def test_quality_gate_rejects_excessive_kl_or_topk_churn():
    metrics = compare_logits([10.0, 9.0, 0.0, -1.0], [0.0, -1.0, 10.0, 9.0], top_k=2)

    violations = logit_quality_gate(metrics, kl_tau=0.01, min_top_k_overlap=0.9)
    assert any("kl_divergence" in violation for violation in violations)
    assert any("top_k_overlap" in violation for violation in violations)


def test_aggregate_gate_uses_mean_tau_and_ten_tau_maximum():
    small = compare_logits([2.0, 1.0, 0.0], [2.01, 0.99, 0.0], top_k=2)
    near_tie = compare_logits([1.0001, 1.0, -2.0], [1.0, 1.0001, -2.0], top_k=2)

    aggregate = aggregate_logit_quality([small, near_tie])

    assert aggregate.sample_count == 2
    assert aggregate.argmax_changes == 1
    assert aggregate.min_top_k_overlap == pytest.approx(1.0)
    assert aggregate_logit_quality_gate(aggregate, kl_tau=0.01) == []


def test_aggregate_gate_fails_closed_on_no_samples():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_logit_quality([])
