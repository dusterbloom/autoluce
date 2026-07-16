"""
Tests for the statistical core (autoluce.bench.statistics) and its wiring into
the significance gates that decide keep/revert and frontier claims.

The t-distribution checks use published t-table quantiles; the multiple-comparison
checks use hand-computed Holm and Benjamini-Hochberg adjustments.
"""

import pytest

from autoluce.bench.statistics import (
    abba_block_deltas,
    benjamini_hochberg,
    geometric_mean_with_stddev,
    holm_adjust,
    one_sample_t_test,
    standard_error,
    student_t_sf,
    welch_t_test,
)
from autoluce.bench.uncertainty import is_significant_improvement_samples


# --- standard error -----------------------------------------------------------


def test_standard_error_matches_definition():
    # samples [1..5]: sample stdev sqrt(2.5), n 5 -> SEM sqrt(2.5)/sqrt(5) = sqrt(0.5)
    assert standard_error([1, 2, 3, 4, 5]) == pytest.approx(0.5 ** 0.5)


def test_standard_error_degenerate_inputs():
    assert standard_error([]) == 0.0
    assert standard_error([42.0]) == 0.0


# --- Student's t survival function against published table values ---------------


@pytest.mark.parametrize(
    ("t", "df", "expected_sf"),
    [
        (0.0, 10, 0.5),
        (2.776, 4, 0.025),    # t_{0.975}(4)
        (2.262, 9, 0.025),    # t_{0.975}(9)
        (3.250, 9, 0.005),    # t_{0.995}(9)
        (2.093, 19, 0.025),   # t_{0.975}(19)
        (2.045, 29, 0.025),   # t_{0.975}(29)
        (-2.776, 4, 0.975),   # symmetry
    ],
)
def test_student_t_sf_matches_t_table(t, df, expected_sf):
    assert student_t_sf(t, df) == pytest.approx(expected_sf, abs=1e-3)


# --- Welch's t-test ------------------------------------------------------------


def test_welch_clear_separation_is_significant():
    result = welch_t_test([10.0, 10.2, 9.8, 10.1], [9.0, 9.2, 8.8, 9.1])
    assert result.p_value < 0.001
    assert result.statistic > 0
    assert result.effect == pytest.approx(1.0)


def test_welch_overlapping_noise_is_not_significant():
    result = welch_t_test([10.0, 10.5, 9.5], [9.8, 10.3, 9.7])
    assert result.p_value > 0.05


def test_welch_zero_variance_is_deterministic_comparison():
    assert welch_t_test([10.0, 10.0, 10.0], [9.0, 9.0, 9.0]).p_value == 0.0
    assert welch_t_test([9.0, 9.0, 9.0], [10.0, 10.0, 10.0]).p_value == 1.0
    assert welch_t_test([10.0, 10.0, 10.0], [10.0, 10.0, 10.0]).p_value == 1.0


def test_welch_requires_variance_from_both_sides():
    with pytest.raises(ValueError):
        welch_t_test([10.0], [9.0, 9.1])
    with pytest.raises(ValueError):
        welch_t_test([10.0, 10.1], [])


# --- paired (ABBA) blocks -------------------------------------------------------


def test_abba_block_deltas_accepts_both_orders_within_pair():
    sequence = [("clean", 100.0), ("candidate", 105.0), ("candidate", 106.0), ("clean", 101.0)]
    assert abba_block_deltas(sequence) == [5.0, 5.0]
    assert abba_block_deltas([("candidate", 105.0), ("clean", 100.0)]) == [5.0]


def test_abba_block_deltas_rejects_malformed_sequences():
    with pytest.raises(ValueError):
        abba_block_deltas([("clean", 100.0), ("clean", 101.0)])
    with pytest.raises(ValueError):
        abba_block_deltas([("clean", 100.0), ("candidate", 105.0), ("clean", 101.0)])
    with pytest.raises(ValueError):
        abba_block_deltas([("clean", 100.0), ("other", 105.0)])


def test_one_sample_t_test_on_block_deltas():
    assert one_sample_t_test([4.8, 5.2, 5.1]).p_value < 0.01
    assert one_sample_t_test([-1.0, 1.0, -0.5, 0.5]).p_value == pytest.approx(0.5)
    with pytest.raises(ValueError):
        one_sample_t_test([5.0])


# --- multiple-comparison corrections --------------------------------------------


def test_holm_adjust_step_down():
    # sorted: 0.01 -> 3*0.01=0.03; 0.03 -> 2*0.03=0.06; 0.04 -> 1*0.04=0.04,
    # monotone envelope lifts the last to 0.06.
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert holm_adjust([0.04]) == pytest.approx([0.04])


def test_benjamini_hochberg_step_up():
    # sorted: rank1 3/1*0.01=0.03; rank2 min(0.04, 3/2*0.03)=0.04; rank3 3/3*0.04=0.04.
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert benjamini_hochberg([0.001, 0.02, 0.03]) == pytest.approx([0.003, 0.03, 0.03])
    assert benjamini_hochberg([]) == []


# --- geometric-mean uncertainty propagation -------------------------------------


def test_geometric_mean_with_stddev_propagates_on_log_scale():
    gm, sigma = geometric_mean_with_stddev([100.0, 200.0], [10.0, 20.0])
    assert gm == pytest.approx((100.0 * 200.0) ** 0.5)
    # sigma = gm * sqrt((10/100)^2 + (20/200)^2) / 2 = gm * 0.0707...
    assert sigma == pytest.approx(10.0)


def test_geometric_mean_with_stddev_degenerate_inputs():
    assert geometric_mean_with_stddev([], []) == (0.0, 0.0)
    assert geometric_mean_with_stddev([0.0, 200.0], [1.0, 2.0]) == (0.0, 0.0)
    gm, sigma = geometric_mean_with_stddev([100.0, 200.0], [0.0, 0.0])
    assert gm == pytest.approx((100.0 * 200.0) ** 0.5)
    assert sigma == 0.0


# --- sample-aware significance gate ----------------------------------------------


def test_samples_gate_clear_win_and_overlap():
    assert is_significant_improvement_samples([10.0, 10.2, 9.8, 10.1], [9.0, 9.2, 8.8, 9.1]) is True
    assert is_significant_improvement_samples([10.0, 10.5, 9.5], [9.8, 10.3, 9.7]) is False


def test_samples_gate_degenerate_falls_back_to_strict_mean_comparison():
    assert is_significant_improvement_samples([110.0], [100.0]) is True
    assert is_significant_improvement_samples([100.0], [100.0]) is False
    assert is_significant_improvement_samples([], [100.0]) is False


def test_samples_gate_respects_alpha():
    # A marginally significant pair (p ~ 0.03) passes at alpha=0.05, fails at 0.01.
    new = [10.0, 11.0, 10.5, 11.5, 10.2, 10.8]
    ref = [9.5, 10.0, 9.8, 10.2, 9.6, 10.1]
    borderline = welch_t_test(new, ref).p_value
    assert is_significant_improvement_samples(new, ref, alpha=borderline + 0.01) is True
    assert is_significant_improvement_samples(new, ref, alpha=borderline - 0.001) is False


# --- aggregation keeps sample arrays usable --------------------------------------


def test_aggregate_scores_passes_through_single_benchmark_score_samples():
    from autoluce.bench.harness import aggregate_scores

    result = {
        "score": 120.0, "score_stddev": 4.0, "decode_tok_s": 120.0,
        "prefill_tok_s": 2500.0, "peak_mem_GiB": 18.0,
        "score_samples": [118.0, 120.0, 122.0],
    }
    agg = aggregate_scores([result])
    assert agg["score_samples"] == [118.0, 120.0, 122.0]


def test_aggregate_scores_does_not_pool_samples_across_benchmarks():
    from autoluce.bench.harness import aggregate_scores

    results = [
        {"score": 120.0, "score_stddev": 4.0, "decode_tok_s": 120.0,
         "prefill_tok_s": 2500.0, "peak_mem_GiB": 18.0, "score_samples": [118.0, 122.0]},
        {"score": 80.0, "score_stddev": 2.0, "decode_tok_s": 80.0,
         "prefill_tok_s": 1500.0, "peak_mem_GiB": 16.0, "score_samples": [79.0, 81.0]},
    ]
    # Pooling raw samples across different workloads is not iid; must not fabricate.
    assert "score_samples" not in aggregate_scores(results)
