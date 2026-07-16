"""Tests for the read-only WebUI adapter.

The WebUI is a thin viewer over authoritative campaign state owned by
``CampaignStore``. These tests pin that contract: discovery delegates to the
real layout, the route id is the stable ``campaign_id``, status is derived
only from authoritative fields (promotion / evidence presence), and routing
is decided by explicit path predicates rather than recovered from errors.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from autoluce.research_contract import Campaign
from autoluce.web import server as web


EXAMPLE_CAMPAIGN = Path(__file__).resolve().parent.parent / "examples" / "research-campaign.json"


def _write_contract(root: Path, rel: str, *, name: str | None = None) -> tuple[Path, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = json.loads(EXAMPLE_CAMPAIGN.read_text())
    if name:
        contract["name"] = name
    path.write_text(json.dumps(contract, indent=2))
    return path, Campaign.from_dict(contract).campaign_id


def _write_state(
    path: Path,
    campaign_id: str,
    *,
    evidence: int = 0,
    frontier: int = 0,
    promotion: str | None = None,
    stage: str = "explore",
) -> None:
    campaign = json.loads(path.read_text())
    campaign["lifecycle_stage"] = stage
    state = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign": campaign,
        "stage": stage,
        "stage_history": ["observe", "discover", stage],
        "references": [],
        "evidence": [
            {
                "evidence_id": f"ev-{i}",
                "campaign_id": campaign_id,
                "metrics": {"prefill_tok_s": 1000.0 + i},
                "gates": {"correctness": True},
                "system": {},
                "workload": {},
                "provenance": {},
                "uncertainty": {},
                "artifact_hash": f"h{i}",
                "schema_version": 1,
                "feasible": True,
                "compatibility_key": "k",
            }
            for i in range(evidence)
        ],
        "frontier": [f"ev-{i}" for i in range(frontier)],
        "comparisons": [],
        "promotion": promotion,
    }
    path.with_suffix(path.suffix + ".state.json").write_text(json.dumps(state, indent=2))


def _fake_handler(command: str = "GET", path: str = "/"):
    handler = type("FakeHandler", (), {})()
    handler.command = command
    handler.path = path
    handler.status = None
    handler.headers: dict[str, str] = {}
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: setattr(handler, "status", status)
    handler.send_header = lambda key, value: handler.headers.__setitem__(key, value)
    handler.end_headers = lambda: None
    return handler


def test_list_discovers_research_and_benchmark_contracts(tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json", name="demo")
    _write_contract(tmp_path, "benchmarks/suite/campaigns/archived.json", name="archived")

    campaigns = web.list_campaigns(root=tmp_path)

    assert {c["name"] for c in campaigns} == {"demo", "archived"}


def test_summary_id_is_campaign_id_and_stage_from_contract(tmp_path):
    _path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")

    [campaign] = web.list_campaigns(root=tmp_path)

    assert campaign["id"] == campaign_id
    assert campaign["campaign_id"] == campaign_id
    assert campaign["stage"] == "observe"
    assert campaign["status"] == "planned"
    assert campaign["evidence_count"] == 0
    assert campaign["promotion"] is None


def test_authoritative_state_drives_counts_and_status(tmp_path):
    path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    _write_state(path, campaign_id, evidence=3, frontier=1, stage="explore")

    [campaign] = web.list_campaigns(root=tmp_path)

    assert campaign["evidence_count"] == 3
    assert campaign["frontier_count"] == 1
    assert campaign["status"] == "measured"
    assert campaign["stage"] == "explore"


def test_promotion_marks_status_promoted(tmp_path):
    path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    _write_state(path, campaign_id, evidence=2, frontier=1, promotion="ev-0")

    [campaign] = web.list_campaigns(root=tmp_path)

    assert campaign["status"] == "promoted"
    assert campaign["promotion"] == "ev-0"


def test_get_campaign_returns_authoritative_state(tmp_path):
    path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    _write_state(path, campaign_id, evidence=2, frontier=1)

    detail = web.get_campaign(campaign_id, root=tmp_path)

    assert detail is not None
    assert detail["campaign_id"] == campaign_id
    assert detail["campaign"]["objective"]["metric"] == "prefill_tok_s"
    assert len(detail["state"]["evidence"]) == 2
    assert len(detail["state"]["frontier"]) == 1


def test_get_campaign_unknown_id_returns_none(tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")

    assert web.get_campaign("does-not-exist", root=tmp_path) is None


def test_corrupt_contract_is_skipped(tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    bad = tmp_path / ".autoluce/research/broken/campaign.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ not valid json")

    campaigns = web.list_campaigns(root=tmp_path)

    assert len(campaigns) == 1


def test_summary_carries_tree_lineage(tmp_path):
    _write_contract(
        tmp_path, ".autoluce/research/normal-kv-prefill/f16-f16/campaign.json", name="f16"
    )
    _write_contract(tmp_path, "benchmarks/q4km/campaigns/run-a.json", name="run-a")

    by_root = {c["root"]: c for c in web.list_campaigns(root=tmp_path)}
    research = by_root["research"]
    benchmark = by_root["benchmarks"]

    assert research["family"] == "normal-kv-prefill"
    assert research["variant"] == "f16-f16"
    assert benchmark["family"] == "q4km"
    assert benchmark["variant"] == "run-a"


def test_generation_bumps_when_campaign_state_changes(tmp_path):
    import os
    import time

    rel = ".autoluce/research/demo/campaign.json"
    _write_contract(tmp_path, rel)
    before = web.generation(root=tmp_path)

    # An ingest writes the campaign's state file -> the generation must move.
    state_path = tmp_path / rel.replace("campaign.json", "campaign.json.state.json")
    state_path.write_text('{"stage":"explore"}')
    forced = time.time() + 5  # deterministic later mtime (production writes are seconds apart)
    os.utime(state_path, (forced, forced))
    after = web.generation(root=tmp_path)

    assert isinstance(before, int)
    assert after > before


def test_dispatch_version_returns_generation(monkeypatch, tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json", name="demo")
    monkeypatch.setattr(web, "ROOT", tmp_path)

    handler = _fake_handler(path="/api/version")
    web.web_dispatch(handler)

    assert handler.status == 200
    payload = json.loads(handler.wfile.getvalue())
    assert isinstance(payload["generation"], int)


def test_version_route_is_a_webui_request():
    assert web.is_webui_request("/api/version")
    assert not web.is_public_webui("/api/version")


def test_summary_sparkline_is_ordered_objective_series(tmp_path):
    path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    _write_state(path, campaign_id, evidence=3, frontier=1)

    [campaign] = web.list_campaigns(root=tmp_path)

    assert campaign["sparkline"] == [1000.0, 1001.0, 1002.0]


def test_routing_predicates_pin_the_webui_surface():
    assert web.is_webui_request("/")
    assert web.is_webui_request("/static/app.js")
    assert web.is_webui_request("/api/campaigns")
    assert web.is_webui_request("/api/campaigns/abc123")
    assert not web.is_webui_request("/v1/status")
    assert not web.is_webui_request("/v1/agents/status")

    assert web.is_public_webui("/")
    assert web.is_public_webui("/static/style.css")
    assert not web.is_public_webui("/api/campaigns")


def test_dispatch_serves_index():
    handler = _fake_handler(path="/")

    web.web_dispatch(handler)

    assert handler.status == 200
    assert b"<!DOCTYPE html>" in handler.wfile.getvalue()


def test_dispatch_lists_campaigns(monkeypatch, tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json", name="demo")
    monkeypatch.setattr(web, "ROOT", tmp_path)

    handler = _fake_handler(path="/api/campaigns")
    web.web_dispatch(handler)

    assert handler.status == 200
    payload = json.loads(handler.wfile.getvalue())
    assert [c["name"] for c in payload] == ["demo"]


def test_dispatch_campaign_detail(monkeypatch, tmp_path):
    path, campaign_id = _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    _write_state(path, campaign_id, evidence=1, frontier=1)
    monkeypatch.setattr(web, "ROOT", tmp_path)

    handler = _fake_handler(path=f"/api/campaigns/{campaign_id}")
    web.web_dispatch(handler)

    assert handler.status == 200
    payload = json.loads(handler.wfile.getvalue())
    assert payload["campaign_id"] == campaign_id
    assert len(payload["state"]["evidence"]) == 1


def test_dispatch_unknown_campaign_returns_404(monkeypatch, tmp_path):
    _write_contract(tmp_path, ".autoluce/research/demo/campaign.json")
    monkeypatch.setattr(web, "ROOT", tmp_path)

    handler = _fake_handler(path="/api/campaigns/nope")
    web.web_dispatch(handler)

    assert handler.status == 404


def test_dispatch_has_no_agent_status_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "ROOT", tmp_path)

    handler = _fake_handler(path="/api/agents/status")

    with pytest.raises(ValueError):
        web.web_dispatch(handler)


def test_dispatch_rejects_non_get():
    handler = _fake_handler(command="POST", path="/api/campaigns")

    with pytest.raises(ValueError):
        web.web_dispatch(handler)


def test_dispatch_rejects_static_traversal():
    handler = _fake_handler(path="/static/../../etc/passwd")

    with pytest.raises(ValueError):
        web.web_dispatch(handler)
