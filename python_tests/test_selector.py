"""
Tests for selector.rank_by_bottleneck: the pure decision that reorders untried
ROADMAP items so those targeting the active profiling bottleneck come first.

This is the testable core of Move 1 (profile -> Amdahl-weighted ideation). The
trace-parser that produces the bound verdict is separate I/O; this function takes
the bound as given, so it is fully deterministic.
"""

from autoggml.ideation.selector import rank_by_bottleneck

ITEMS = [(3, "self-speculative"), (8, "cuda-graph verify"), (10, "KV rollback"), (11, "draft quant"), (12, "Q4_K vulkan")]
RELEVANCE = {"memory": [10, 11, 14], "compute": [9, 12], "overhead": [8, 9]}


def test_no_bound_preserves_numerical_order():
    ranked = rank_by_bottleneck(ITEMS, None, RELEVANCE)
    assert [n for n, _, _ in ranked] == [3, 8, 10, 11, 12]
    assert not any(m for _, _, m in ranked)


def test_memory_bound_surfaces_memory_items_first_in_numerical_groups():
    ranked = rank_by_bottleneck(ITEMS, "memory", RELEVANCE)
    assert [n for n, _, _ in ranked] == [10, 11, 3, 8, 12]  # matched [10,11] then off [3,8,12]


def test_compute_bound_surfaces_compute_items_first():
    # 9 is not in ITEMS, so only 12 matches; rest stay numerical.
    ranked = rank_by_bottleneck(ITEMS, "compute", RELEVANCE)
    assert [n for n, _, _ in ranked] == [12, 3, 8, 10, 11]


def test_matched_flag_marks_relevance():
    ranked = rank_by_bottleneck(ITEMS, "memory", RELEVANCE)
    by_num = {n: m for n, _, m in ranked}
    assert by_num[10] is True and by_num[11] is True
    assert by_num[3] is False and by_num[8] is False


def test_unknown_bound_falls_back_to_numerical():
    ranked = rank_by_bottleneck(ITEMS, "bogus", RELEVANCE)
    assert [n for n, _, _ in ranked] == [3, 8, 10, 11, 12]


def test_empty_items_returns_empty():
    assert rank_by_bottleneck([], "memory", RELEVANCE) == []


def test_defaults_to_profiling_roadmap_when_relevance_omitted():
    # DRY: the bound->items mapping lives in profiling.ROADMAP_FOR_BOUND.
    ranked = rank_by_bottleneck([(10, "KV"), (3, "off")], "memory")
    assert ranked[0][:2] == (10, "KV") and ranked[0][2] is True
    assert ranked[1][2] is False
