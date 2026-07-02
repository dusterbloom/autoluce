"""
Tests for the living ideas queue: parse ROADMAP.md idea items, read which ROADMAP
numbers have been tried (from experiment descriptions tagged "[#N]"), report
what's left. Advisory — helps the agent pick the next untried idea.
"""

from autoggml.ideation.ideas import extract_tried_numbers, load_roadmap_items, untried


def test_load_roadmap_items_parses_numbered_bold_titles():
    text = """
## Algorithmic
1. **Tree speculative decoding.** Verify a tree of candidates.
2. **Hidden-state drafting.** Draft from hidden states.

some non-item line.
"""
    assert load_roadmap_items(text) == [(1, "Tree speculative decoding"), (2, "Hidden-state drafting")]


def test_extract_tried_numbers_reads_tags():
    descriptions = ["[#3] adaptive K controller", "march=native (no tag)", "[#1] + [#2] combined"]
    assert extract_tried_numbers(descriptions) == {1, 2, 3}


def test_extract_tried_numbers_empty_when_no_tags():
    assert extract_tried_numbers(["baseline", "march=native"]) == set()


def test_untried_returns_items_not_in_tried():
    items = [(1, "A"), (2, "B"), (3, "C")]
    assert untried(items, {2}) == [(1, "A"), (3, "C")]
    assert untried(items, {1, 2, 3}) == []
