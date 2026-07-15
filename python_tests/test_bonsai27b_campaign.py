"""Contract tests for the Bonsai-27B Q1 target and DSpark campaigns."""

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


def test_bonsai_dspark_benchmark_is_frontier_eligible():
    spec = json.loads((ROOT / "benchmarks" / "bonsai27b-q1-dspark.json").read_text())

    assert spec["manifest_entry"] == "bonsai-27b-q1"
    assert spec["spec_type"] == "draft-dspark"
    assert spec["n_draft"] == 4
    assert spec["quality"] == "exact"
    assert spec["frontier_eligible"] is True
    assert spec["contexts"] == [512, 4096, 16384]
    assert spec["expected"] == {
        "min_decode_tok_s": 35.0,
        "min_acceptance_rate": 0.3,
        "max_peak_mem_GiB": 24.0,
    }
