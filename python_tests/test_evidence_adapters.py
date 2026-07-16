"""Tests for the head-to-head producer adapter (step 6)."""

from __future__ import annotations

import json
from pathlib import Path

from autoluce.evidence_adapters import ingest_head_to_head
from autoluce.research import Campaign, CampaignStore

SYSTEM = {
    "machine": "machine-a", "model": "qwen36-27b", "model_fingerprint": "mf",
    "runtime": "lucebox:dflash-server-http", "hardware": "rtx-3090", "backend": "cuda",
    "quantization": "q4_k_m", "environment": "env-a",
}
WORKLOAD = {"contexts": [1024, 8192], "batch_shape": {"batch": 1, "ubatch": 512},
            "mode": "prefill", "prompts": ["synthetic"]}
OBJECTIVE = {"metric": "prefill_tok_s", "direction": "maximize"}
CONSTRAINTS = {"correctness": "numeric-kernel-parity"}


def _store(tmp_path: Path) -> CampaignStore:
    path = tmp_path / "campaign.json"
    campaign = Campaign(name="q4km-vs-llamacpp", system=SYSTEM, workload=WORKLOAD,
                        objective=OBJECTIVE, constraints=CONSTRAINTS)
    path.write_text(json.dumps(campaign.to_dict()))
    return CampaignStore(path)


def test_head_to_head_records_candidate_evidence_and_comparison(tmp_path):
    store = _store(tmp_path)
    rows = [
        {"context": 1024, "metric": "prefill_tok_s", "candidate": 1341.47, "reference": 1278.72, "delta_pct": 4.91},
        {"context": 8192, "metric": "prefill_tok_s", "candidate": 1393.73, "reference": 1359.53, "delta_pct": 2.52},
    ]

    report = ingest_head_to_head(
        store, rows, engine_reference="llama.cpp",
        reference_locator="llama.cpp@00f5442",
    )

    assert report["recorded"] == 2
    state = CampaignStore(store.campaign_path).load()
    assert len(state["evidence"]) == 2                      # one candidate evidence per context
    head_to_head = [c for c in state["comparisons"] if c["kind"] == "head_to_head"]
    assert len(head_to_head) == 2
    by_ctx = {c["context"]: c for c in head_to_head}
    assert by_ctx[1024]["engine_candidate"] == "dFlash"
    assert by_ctx[1024]["engine_reference"] == "llama.cpp"
    assert by_ctx[1024]["delta_pct"] == 4.91
    assert state["stage"] == "compare"                      # advanced by the comparison
