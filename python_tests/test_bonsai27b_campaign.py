"""Contract tests for the initial Bonsai-27B Q1 target-only campaign."""

import json

from autoluce import ROOT


def test_bonsai_target_benchmark_starts_as_a_non_frontier_canary():
    spec = json.loads((ROOT / "benchmarks" / "bonsai27b-q1-target.json").read_text())

    assert spec["manifest_entry"] == "bonsai-27b-q1"
    assert spec["spec_type"] == "target-only"
    assert spec["n_draft"] == 0
    assert spec["quality"] == "canary"
    assert spec["frontier_eligible"] is False
    assert spec["contexts"] == [512, 4096, 16384]
    assert spec["prompt_token_reserve"] == 32
    assert spec["objective"] == {
        "maximize": "decode_tok_s",
        "constraints": {"peak_mem_GiB": {"max": 24.0}},
    }
