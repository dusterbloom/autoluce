"""
Tests for propose.build_proposal_messages: the pure rendering of harness state into a
chat prompt. main() is a thin CLI (read state -> build -> complete -> print) and is
exercised by the no-op-when-disabled smoke, not unit-tested here.
"""

from autoluce.ideation.propose import build_proposal_messages


def test_proposal_messages_include_best_score_and_ideas():
    msgs = build_proposal_messages([(10, "KV rollback"), (11, "draft quant")], best_score=123.4)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "123.4" in msgs[1]["content"]
    assert "#10. KV rollback" in msgs[1]["content"]
    assert "#11. draft quant" in msgs[1]["content"]


def test_proposal_messages_note_the_bound_when_given():
    msgs = build_proposal_messages([(9, "fuse draft")], best_score=50.0, bound="compute")
    assert "compute" in msgs[1]["content"]


def test_proposal_messages_omit_bound_note_when_absent():
    msgs = build_proposal_messages([(9, "fuse draft")], best_score=50.0)
    assert "compute" not in msgs[1]["content"]


def test_proposal_messages_handle_empty_idea_list():
    msgs = build_proposal_messages([], best_score=0.0)
    # Must not crash, and must signal there's nothing queued.
    assert "none" in msgs[1]["content"].lower()
