"""
Synthetic-data tests for the constrained objective (objective.check_constraints).
"""

import pytest

from autoggml.bench.objective import check_constraints


def _spec(constraints: dict) -> dict:
    return {"objective": {"maximize": "decode_tok_s", "constraints": constraints}}


def test_empty_spec_passes():
    assert check_constraints({"decode_tok_s": 100.0}, None, {}, k=1.0) == []


def test_spec_without_constraints_block_passes():
    spec = {"objective": {"maximize": "decode_tok_s"}}
    assert check_constraints({"decode_tok_s": 100.0}, None, spec, k=1.0) == []


def test_max_bound_passes_at_exactly_k_sigma():
    spec = _spec({"peak_mem_GiB": {"max": 22.0}})
    metrics = {"peak_mem_GiB": 20.0, "peak_mem_GiB_stddev": 2.0}
    assert check_constraints(metrics, None, spec, k=1.0) == []  # 20 + 1*2 == 22


def test_max_bound_fails_just_beyond_k_sigma():
    spec = _spec({"peak_mem_GiB": {"max": 22.0}})
    metrics = {"peak_mem_GiB": 20.1, "peak_mem_GiB_stddev": 2.0}
    violations = check_constraints(metrics, None, spec, k=1.0)
    assert len(violations) == 1
    assert "peak_mem_GiB" in violations[0]


def test_min_bound_passes_at_exactly_k_sigma_and_fails_below():
    spec = _spec({"prefill_tok_s": {"min": 95.0}})
    ok = {"prefill_tok_s": 100.0, "prefill_tok_s_stddev": 5.0}
    bad = {"prefill_tok_s": 99.9, "prefill_tok_s_stddev": 5.0}
    assert check_constraints(ok, None, spec, k=1.0) == []  # 100 - 1*5 == 95
    assert check_constraints(bad, None, spec, k=1.0) != []


def test_missing_stddev_treated_as_zero():
    spec = _spec({"peak_mem_GiB": {"max": 22.0}})
    assert check_constraints({"peak_mem_GiB": 22.0}, None, spec, k=3.0) == []
    assert check_constraints({"peak_mem_GiB": 22.01}, None, spec, k=3.0) != []


def test_k_scales_the_margin():
    spec = _spec({"peak_mem_GiB": {"max": 22.0}})
    metrics = {"peak_mem_GiB": 20.0, "peak_mem_GiB_stddev": 1.5}
    assert check_constraints(metrics, None, spec, k=1.0) == []   # 21.5 <= 22
    assert check_constraints(metrics, None, spec, k=2.0) != []   # 23.0 > 22


def test_min_frac_of_baseline_uses_baseline_value():
    spec = _spec({"prefill_tok_s": {"min_frac_of_baseline": 0.95}})
    baseline = {"prefill_tok_s": 1000.0}
    assert check_constraints({"prefill_tok_s": 950.0}, baseline, spec, k=1.0) == []
    assert check_constraints({"prefill_tok_s": 949.0}, baseline, spec, k=1.0) != []


def test_min_frac_of_baseline_without_baseline_raises():
    spec = _spec({"prefill_tok_s": {"min_frac_of_baseline": 0.95}})
    with pytest.raises(ValueError):
        check_constraints({"prefill_tok_s": 950.0}, None, spec, k=1.0)


def test_constrained_metric_missing_from_metrics_is_a_violation():
    spec = _spec({"peak_mem_GiB": {"max": 22.0}})
    assert check_constraints({"decode_tok_s": 100.0}, None, spec, k=1.0) != []


def test_unknown_constraint_form_raises():
    spec = _spec({"peak_mem_GiB": {"median": 22.0}})
    with pytest.raises(ValueError):
        check_constraints({"peak_mem_GiB": 1.0}, None, spec, k=1.0)
