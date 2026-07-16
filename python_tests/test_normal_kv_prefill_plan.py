from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoluce.prefill_plan import (
    NORMAL_KV_LANES,
    PREFILL_HYPOTHESES,
    campaign_for_kv_lane,
    measurement_cells,
    require_promotion_measurement,
)
from autoluce.research_contract import Campaign
from autoluce.research_evidence import CampaignEvidence, CompatibilityError
from autoluce.research import CampaignStore


ROOT = Path(__file__).resolve().parents[1]


def _template() -> Campaign:
    return Campaign(
        name="rtx3090-qwen36-q4km-prefill",
        system={
            "machine": "machine-3090",
            "model": "qwen36-27b",
            "model_fingerprint": "model-q4km",
            "runtime": "lucebox:dflash-server-http",
            "hardware": "rtx-3090-sm86",
            "backend": "cuda",
            "quantization": "Q4_K_M",
            "environment": "cuda-12.6-driver-pinned",
        },
        workload={
            "contexts": [1024, 8192, 16384, 65536],
            "batch_shape": {"batch": 1, "ubatch": 512},
            "mode": "prefill",
            "prompts": ["Continue analyzing this deterministic synthetic record."],
        },
        objective={"metric": "prefill_tok_s", "direction": "maximize"},
        constraints={
            "correctness": "exact",
            "peak_mem_GiB": {"max": 24.0},
            "evidence_profile": "normal_kv_prefill_v1",
        },
    )


def _evidence(campaign: Campaign, *, runtime: str | None = None) -> CampaignEvidence:
    system = dict(campaign.system)
    if runtime is not None:
        system["runtime"] = runtime
    return CampaignEvidence.create(
        campaign_id=campaign.campaign_id,
        system=system,
        workload=campaign.workload,
        metrics={"prefill_tok_s": 1400.0},
        gates={"correctness": True},
        artifact_hash="sha256:binary",
    )


def test_normal_lanes_are_explicit_and_exclude_tq3():
    assert [lane.name for lane in NORMAL_KV_LANES] == ["f16-f16", "q8_0-q8_0", "q4_0-q4_0"]
    assert all("tq3" not in lane.key_type and "tq3" not in lane.value_type for lane in NORMAL_KV_LANES)

    with pytest.raises(ValueError, match="normal KV campaign"):
        campaign_for_kv_lane(_template(), "tq3_0-tq3_0")


def test_each_lane_has_an_explicit_runtime_pair_and_distinct_campaign_identity():
    campaigns = [campaign_for_kv_lane(_template(), lane.name) for lane in NORMAL_KV_LANES]

    assert len({campaign.campaign_id for campaign in campaigns}) == len(campaigns)
    assert {campaign.system["quantization"]["weights"] for campaign in campaigns} == {"Q4_K_M"}
    assert [campaign.system["quantization"]["kv_cache"] for campaign in campaigns] == [
        {"key": "f16", "value": "f16"},
        {"key": "q8_0", "value": "q8_0"},
        {"key": "q4_0", "value": "q4_0"},
    ]
    assert [lane.runtime_env for lane in NORMAL_KV_LANES] == [
        {"DFLASH27B_KV_K": "f16", "DFLASH27B_KV_V": "f16"},
        {"DFLASH27B_KV_K": "q8_0", "DFLASH27B_KV_V": "q8_0"},
        {"DFLASH27B_KV_K": "q4_0", "DFLASH27B_KV_V": "q4_0"},
    ]
    assert all("runtime_env" not in campaign.workload for campaign in campaigns)


def test_cross_lane_evidence_is_not_comparable_but_runtime_reference_is():
    f16 = campaign_for_kv_lane(_template(), "f16-f16")
    q8 = campaign_for_kv_lane(_template(), "q8_0-q8_0")

    with pytest.raises(CompatibilityError, match="quantization"):
        _evidence(f16).compare(_evidence(q8))

    comparison = _evidence(f16).compare(
        _evidence(f16, runtime="upstream-llama"),
        allowed_system_variations=frozenset({"runtime"}),
    )
    assert comparison["compatible"] is True


def test_measurement_matrix_marks_128k_as_a_fit_probe_not_frontier_evidence():
    f16_cells = measurement_cells("f16-f16")
    q8_cells = measurement_cells("q8_0-q8_0")

    assert [cell.context for cell in f16_cells] == [1024, 8192, 16384, 65536]
    assert all(cell.frontier_eligible for cell in f16_cells)
    assert [cell.context for cell in q8_cells] == [1024, 8192, 16384, 65536, 131072]
    assert q8_cells[-1].fit_probe is True
    assert q8_cells[-1].frontier_eligible is False


def test_hypotheses_are_ordered_and_change_one_policy_at_a_time():
    assert [item.name for item in PREFILL_HYPOTHESES] == [
        "explicit-baseline",
        "attention-dispatch",
        "gdn-chunking",
        "causal-mask",
        "quantized-attention",
        "depth-schedule",
        "graph-cleanup",
    ]
    assert all(len(item.controls) == 1 for item in PREFILL_HYPOTHESES[1:])
    assert all("tq3" not in item.name for item in PREFILL_HYPOTHESES)


def test_promotion_measurement_requires_statistics_quality_and_resolved_kv_provenance():
    measurement = {
        "metrics": {
            "prefill_tok_s": 1450.0,
            "prefill_tok_s_stddev": 4.0,
            "prefill_tok_s_samples": [1445.0, 1450.0, 1455.0],
            "peak_mem_GiB": 21.5,
        },
        "quality": {"correctness": {"kind": "exact", "passed": True}},
        "provenance": {
            "binary_sha256": "binary-a",
            "product_digest": "product-a",
            "model_fingerprint": "model-q4km",
            "machine_fingerprint": "machine-3090",
            "resolved_kv_cache": {"key": "q8_0", "value": "q8_0"},
            "context_depth": 65536,
            "prompt_tokens": 65498,
        },
    }

    require_promotion_measurement(measurement, "q8_0-q8_0", context=65536)

    bad = {
        **measurement,
        "provenance": {**measurement["provenance"], "resolved_kv_cache": {"key": "q4_0", "value": "q4_0"}},
    }
    with pytest.raises(ValueError, match="resolved KV cache"):
        require_promotion_measurement(bad, "q8_0-q8_0", context=65536)


def test_regular_weight_prefill_benchmark_is_target_only_and_long_context():
    benchmark = json.loads((ROOT / "benchmarks/qwen36-27b-prefill.json").read_text())

    assert benchmark["manifest_entry"] == "qwen36-27b"
    assert benchmark["spec_type"] == "target-only"
    assert benchmark["contexts"] == [1024, 8192, 16384, 65536]
    assert benchmark["llama_bench_args"]["-n"] == 1
    assert benchmark["objective"]["maximize"] == "prefill_tok_s"


def _result_bundle(*, include_samples: bool) -> dict:
    cells = []
    for context in (1024, 8192, 16384, 65536):
        cell = {
            "context_depth": context,
            "prompt_tokens": context - 38,
            "prefill_tok_s": 1450.0,
            "prefill_tok_s_stddev": 4.0,
            "peak_mem_GiB": 21.5,
        }
        if include_samples:
            cell["prefill_tok_s_samples"] = [1445.0, 1450.0, 1455.0]
        cells.append(cell)
    return {
        "metrics": {
            "prefill_tok_s": 1450.0,
            "prefill_tok_s_stddev": 4.0,
            "peak_mem_GiB": 21.5,
        },
        "quality": {"correctness": {"kind": "exact", "passed": True}},
        "provenance": {
            "binary_sha256": "binary-a",
            "product_digest": "product-a",
            "model_fingerprint": "model-q4km",
            "machine_fingerprint": "machine-3090",
            "resolved_kv_cache": {"key": "q8_0", "value": "q8_0"},
        },
        "benchmarks": [{"benchmark": "qwen36-27b-prefill", "context_metrics": cells}],
    }


def test_incomplete_normal_kv_bundle_is_archived_but_cannot_enter_the_frontier(tmp_path):
    campaign = campaign_for_kv_lane(_template(), "q8_0-q8_0")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign.to_dict()))
    store = CampaignStore(path)

    state = store.load()
    evidence = store.record(state, _result_bundle(include_samples=False))

    assert evidence.gates["promotion_evidence"] is False
    assert state["frontier"] == []
    assert "context 1024" in evidence.provenance["promotion_evidence_violations"][0]


def test_complete_normal_kv_bundle_can_enter_the_frontier(tmp_path):
    campaign = campaign_for_kv_lane(_template(), "q8_0-q8_0")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign.to_dict()))
    store = CampaignStore(path)

    state = store.load()
    evidence = store.record(state, _result_bundle(include_samples=True))

    assert evidence.gates["promotion_evidence"] is True
    assert state["frontier"] == [evidence.evidence_id]


def test_example_campaigns_are_the_same_contract_with_distinct_kv_lanes():
    examples = ROOT / "examples/normal-kv-prefill"
    loaded = [
        Campaign.from_dict(json.loads((examples / f"qwen36-q4km-prefill-{lane.name}.json").read_text()))
        for lane in NORMAL_KV_LANES
    ]

    assert [campaign.campaign_id for campaign in loaded] == [
        campaign_for_kv_lane(_template(), lane.name).campaign_id for lane in NORMAL_KV_LANES
    ]
    assert all(campaign.reference is None for campaign in loaded)
    assert all(campaign.lifecycle_stage == "observe" for campaign in loaded)
