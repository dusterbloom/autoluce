"""Foundation tests: CampaignStore as a locked, crash-safe evidence log.

These pin the corrected durability model (the codex-reviewed design):
  - ingest() is ONE locked transaction (publish immutable -> update state); no
    concurrent-writer lost updates.
  - immutable publication is crash-safe and idempotent (content-addressed), so a
    crash between publish and state-update recovers on retry.
  - state.json is a rebuildable index over the evidence directory, not the sole
    truth: verify() flags orphans/corruption, rebuild_index() restores it.
  - the same rebuild path hydrates stateless benchmark archives from their
    suite-level evidence/ directory.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from autoluce.research import Campaign, CampaignEvidence, CampaignStore


SYSTEM = {
    "machine": "machine-a",
    "model": "qwen3.6-27b",
    "model_fingerprint": "model-a",
    "runtime": "lucebox:dflash-server-http",
    "hardware": "rtx-3090",
    "backend": "cuda",
    "quantization": "iq4_xs",
    "environment": "env-a",
}
WORKLOAD = {
    "contexts": [8192],
    "batch_shape": {"batch": 1, "ubatch": 1},
    "mode": "prefill",
    "prompts": ["deterministic synthetic prompt"],
}
OBJECTIVE = {"metric": "prefill_tok_s", "direction": "maximize"}
CONSTRAINTS = {"correctness": "exact"}


def _write_campaign(path: Path) -> str:
    campaign = Campaign(
        name="foundation", system=SYSTEM, workload=WORKLOAD,
        objective=OBJECTIVE, constraints=CONSTRAINTS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign.to_dict()))
    return campaign.campaign_id


def _raw(value: float = 1400.0, artifact: str = "sha256:a") -> dict:
    return {
        "metrics": {"prefill_tok_s": value},
        "quality": {"correctness": {"kind": "exact", "passed": True}},
        "artifact_hash": artifact,
        "provenance": {},
    }


def _store(tmp_path: Path, rel: str = "campaign.json") -> CampaignStore:
    path = tmp_path / rel
    _write_campaign(path)
    return CampaignStore(path)


# --- ingest: one locked, persisted transaction --------------------------------

def test_ingest_persists_state_without_manual_save(tmp_path):
    store = _store(tmp_path)
    evidence = store.ingest(_raw(1400.0))

    # A fresh reader sees the evidence without the caller ever calling save().
    state = CampaignStore(store.campaign_path).load()
    assert [e["evidence_id"] for e in state["evidence"]] == [evidence.evidence_id]
    assert (store.evidence_dir / f"{evidence.evidence_id}.json").exists()


def test_concurrent_ingests_keep_every_record(tmp_path):
    store = _store(tmp_path)
    recorded: list[str] = []

    def worker(value: float) -> None:
        recorded.append(store.ingest(_raw(value, artifact=f"sha256:{value}")).evidence_id)

    threads = [threading.Thread(target=worker, args=(v,)) for v in (1400.0, 1500.0, 1600.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state = CampaignStore(store.campaign_path).load()
    state_ids = {e["evidence_id"] for e in state["evidence"]}
    assert set(recorded) <= state_ids
    assert len(state["evidence"]) == 3


def test_ingest_is_idempotent_after_state_loss(tmp_path):
    store = _store(tmp_path)
    evidence = store.ingest(_raw(1400.0))

    # Simulate a crash that lost the state index but left the published artifact.
    Path(store.state_path).unlink()

    replayed = store.ingest(_raw(1400.0))
    assert replayed.evidence_id == evidence.evidence_id
    state = CampaignStore(store.campaign_path).load()
    assert len(state["evidence"]) == 1


# --- state as a rebuildable index --------------------------------------------

def test_rebuild_index_recovers_orphaned_evidence(tmp_path):
    store = _store(tmp_path)
    evidence = store.ingest(_raw(1400.0))

    # Someone (or a crash) emptied the index while the artifact remains on disk.
    state = store.load()
    state["evidence"] = []
    state["frontier"] = []
    store.save(state)

    report = store.rebuild_index()
    assert [e["evidence_id"] for e in report["recovered"]] == [evidence.evidence_id]

    restored = CampaignStore(store.campaign_path).load()
    assert [e["evidence_id"] for e in restored["evidence"]] == [evidence.evidence_id]


def test_verify_flags_tampered_evidence(tmp_path):
    store = _store(tmp_path)
    evidence = store.ingest(_raw(1400.0))

    path = store.evidence_dir / f"{evidence.evidence_id}.json"
    payload = json.loads(path.read_text())
    payload["metrics"]["prefill_tok_s"] = 9999.0  # content no longer matches the id
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    report = store.verify()
    assert evidence.evidence_id in report["corrupt"]


# --- archive hydration from the suite-level evidence directory ----------------

def test_rebuild_index_hydrates_archive_from_suite_evidence(tmp_path):
    campaign_path = tmp_path / "benchmarks" / "suite" / "campaigns" / "archived.json"
    campaign_id = _write_campaign(campaign_path)
    suite_evidence = campaign_path.parent.parent / "evidence"
    suite_evidence.mkdir(parents=True, exist_ok=True)

    evidence = CampaignEvidence.create(
        campaign_id=campaign_id, system=SYSTEM, workload=WORKLOAD,
        metrics={"prefill_tok_s": 1400.0}, gates={"quality": True},
        artifact_hash="sha256:a", uncertainty={}, provenance={},
    )
    (suite_evidence / f"{evidence.evidence_id}.json").write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True)
    )

    store = CampaignStore(campaign_path)
    report = store.rebuild_index()
    assert [e["evidence_id"] for e in report["recovered"]] == [evidence.evidence_id]

    state = store.load()
    assert [e["evidence_id"] for e in state["evidence"]] == [evidence.evidence_id]


def test_rebuild_index_excludes_foreign_campaign_evidence(tmp_path):
    # Two campaigns share one suite-level evidence dir; each must index only its own lane.
    base = tmp_path / "benchmarks" / "suite" / "campaigns"
    base.mkdir(parents=True)
    campaign_a = Campaign(name="a", system=SYSTEM, workload={**WORKLOAD, "contexts": [8192]},
                          objective=OBJECTIVE, constraints=CONSTRAINTS)
    campaign_b = Campaign(name="b", system=SYSTEM, workload={**WORKLOAD, "contexts": [32768]},
                          objective=OBJECTIVE, constraints=CONSTRAINTS)
    (base / "a.json").write_text(json.dumps(campaign_a.to_dict()))
    (base / "b.json").write_text(json.dumps(campaign_b.to_dict()))
    suite_evidence = base.parent / "evidence"
    suite_evidence.mkdir(parents=True)

    def seed(campaign_id: str, value: float) -> str:
        ev = CampaignEvidence.create(
            campaign_id=campaign_id, system=SYSTEM, workload=WORKLOAD,
            metrics={"prefill_tok_s": value}, gates={"quality": True},
            artifact_hash=f"sha256:{value}", uncertainty={}, provenance={},
        )
        (suite_evidence / f"{ev.evidence_id}.json").write_text(
            json.dumps(ev.to_dict(), indent=2, sort_keys=True)
        )
        return ev.evidence_id

    id_a = seed(campaign_a.campaign_id, 1400.0)
    id_b = seed(campaign_b.campaign_id, 1500.0)

    report = CampaignStore(base / "a.json").rebuild_index()
    recovered = {e["evidence_id"] for e in report["recovered"]}
    assert recovered == {id_a}
    assert id_b not in recovered


def test_rebuild_index_recovers_from_corrupt_state(tmp_path):
    store = _store(tmp_path)
    evidence = store.ingest(_raw(1400.0))

    # Tear the state index (e.g. a crash left a half-written file).
    Path(store.state_path).write_text("{ torn json")

    report = store.rebuild_index()
    assert [e["evidence_id"] for e in report["recovered"]] == [evidence.evidence_id]
    state = CampaignStore(store.campaign_path).load()
    assert [e["evidence_id"] for e in state["evidence"]] == [evidence.evidence_id]
