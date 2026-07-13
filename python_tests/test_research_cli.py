"""Acceptance contract for the single-file ``autoluce research`` workflow."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path

from cli import resolve


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _campaign():
    return {
        "schema_version": 1,
        "name": "local-prefill",
        "system_under_test": {
            "model": "qwen36-27b",
            "model_fingerprint": "model-a",
            "runtime": "lucebox",
            "hardware": "rtx3090",
            "backend": "cuda",
            "quantization": "IQ4_XS",
        },
        "workload": {
            "contexts": [1024, 8192],
            "batch_shape": {"batch": 1, "ubatch": 512},
            "mode": "prefill",
            "prompts": ["deterministic prompt"],
        },
        "objective": {"metric": "prefill_tok_s", "direction": "maximize"},
        "constraints": [{"metric": "correctness", "operator": "==", "value": "pass"}],
    }


def _evidence():
    return {
        "metrics": {"prefill_tok_s": 1250.0, "prefill_tok_s_stddev": 10.0},
        "quality": {"correctness": "pass"},
        "provenance": {"product_digest": "product-a", "binary_sha256": "binary-a"},
    }


def _reference(hardware="rtx3090"):
    workload = _campaign()["workload"]
    provenance = _evidence()["provenance"]
    digest = lambda value: hashlib.sha256(  # noqa: E731 - compact fixture identity helper
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "kind": "runtime",
        "name": "upstream-llama" if hardware == "rtx3090" else "other-machine",
        "measurement": {"prefill_tok_s": 1200.0, "prefill_tok_s_stddev": 8.0},
        "compatibility": {
            "machine": "rtx3090",
            "model": "qwen36-27b",
            "model_fingerprint": "model-a",
            "hardware": hardware,
            "backend": "cuda",
            "quantization": "IQ4_XS",
            "environment": f"environment-{digest(provenance)}",
            "workload": "local-prefill",
            "workload_fingerprint": f"workload-{digest(workload)}",
        },
    }


def _run(capsys, argv):
    research = importlib.import_module("autoluce.research")
    result = research.main(argv)
    captured = capsys.readouterr()
    return result, json.loads(captured.out) if captured.out.strip() else None, captured.err


def test_research_is_a_thin_cli_route():
    assert resolve(["research", "--json"]) == ("autoluce.research", ["--json"])


def test_no_reference_create_and_show_are_json_capable(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    _write(campaign, _campaign())

    rc, created, _ = _run(capsys, ["--campaign", str(campaign), "--json"])
    assert rc == 0
    assert created["stage"] == "observe"
    assert created["references"] == []

    rc, shown, _ = _run(capsys, ["--campaign", str(campaign), "--json"])
    assert rc == 0
    assert shown == created


def test_reference_can_attach_later_without_rewriting_evidence_ids(tmp_path, capsys, monkeypatch):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    references = tmp_path / "references"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _write(references / "upstream-llama.json", _reference())
    monkeypatch.setenv("AUTOLUCE_REFERENCE_DIR", str(references))

    _, recorded, _ = _run(capsys, ["--campaign", str(campaign), "--record", str(evidence), "--json"])
    evidence_ids = [item["evidence_id"] for item in recorded["evidence"]]

    _, with_goal, _ = _run(
        capsys, ["--campaign", str(campaign), "--goal", "prefill_tok_s >= 1500", "--json"],
    )
    _, with_runtime, _ = _run(
        capsys, ["--campaign", str(campaign), "--against", "upstream-llama", "--json"],
    )
    assert [item["evidence_id"] for item in with_goal["evidence"]] == evidence_ids
    assert [item["evidence_id"] for item in with_runtime["evidence"]] == evidence_ids
    assert {item["kind"] for item in with_runtime["references"]} == {"goal", "runtime"}


def test_compare_refuses_incompatible_evidence(tmp_path, capsys, monkeypatch):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    references = tmp_path / "references"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _write(references / "other-machine.json", _reference(hardware="strix-halo"))
    monkeypatch.setenv("AUTOLUCE_REFERENCE_DIR", str(references))
    _run(capsys, ["--campaign", str(campaign), "--record", str(evidence), "--json"])
    for stage in ("discover", "explore"):
        _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])

    rc, payload, stderr = _run(
        capsys,
        ["--campaign", str(campaign), "--against", "other-machine", "--compare", "--json"],
    )
    assert rc != 0
    assert "incompatible" in ((payload or {}).get("error", "") + stderr).lower()
    assert "hardware" in ((payload or {}).get("error", "") + stderr).lower()


def test_compare_is_optional_in_stage_progression(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    _write(campaign, _campaign())

    for stage in ("discover", "explore", "explain"):
        rc, state, _ = _run(
            capsys, ["--campaign", str(campaign), "--advance", stage, "--json"],
        )
        assert rc == 0
        assert state["stage"] == stage
    assert "compare" not in state["stage_history"]


def test_advance_cannot_claim_promote_without_an_evidence_id(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    _write(campaign, _campaign())
    for stage in ("discover", "explore", "explain"):
        _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])

    rc, payload, _ = _run(capsys, ["--campaign", str(campaign), "--advance", "promote", "--json"])
    assert rc != 0
    assert "--promote" in payload["error"]


def test_promoted_baseline_can_start_a_new_cycle_and_gate_a_successor(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    spec = _campaign()
    spec["constraints"] = {
        "gates": [{
            "metric": "accepted_baseline_fraction",
            "operator": ">=",
            "value": 1.05,
            "verification": "comparison_required",
        }],
    }
    _write(campaign, spec)
    _write(first, {**_evidence(), "metrics": {"prefill_tok_s": 1000.0}})
    _write(second, {**_evidence(), "metrics": {"prefill_tok_s": 1100.0}})

    _, state, _ = _run(capsys, ["--campaign", str(campaign), "--record", str(first), "--json"])
    first_id = state["frontier"][0]
    for stage in ("discover", "explore", "explain"):
        _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])
    rc, _, _ = _run(capsys, ["--campaign", str(campaign), "--promote", first_id, "--json"])
    assert rc == 0

    _run(capsys, ["--campaign", str(campaign), "--record", str(second), "--json"])
    for stage in ("discover", "explore"):
        rc, _, _ = _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])
        assert rc == 0
    rc, compared, _ = _run(
        capsys,
        ["--campaign", str(campaign), "--against", "baseline", "--compare", "--json"],
    )
    assert rc == 0
    assert compared["comparisons"][-1]["accepted_baseline_fraction"] == 1.1
    second_id = compared["frontier"][0]
    _run(capsys, ["--campaign", str(campaign), "--advance", "explain", "--json"])
    rc, promoted, _ = _run(
        capsys, ["--campaign", str(campaign), "--promote", second_id, "--json"],
    )
    assert rc == 0
    assert promoted["promotion"] == second_id
    assert promoted["stage_history"].count("discover") == 2


def test_advancing_to_compare_requires_an_attached_reference(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    _write(campaign, _campaign())
    for stage in ("discover", "explore"):
        rc, _, _ = _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])
        assert rc == 0

    rc, payload, _ = _run(capsys, ["--campaign", str(campaign), "--advance", "compare", "--json"])
    assert rc != 0
    assert "reference" in payload["error"].lower()
    _, state, _ = _run(capsys, ["--campaign", str(campaign), "--json"])
    assert state["stage"] == "explore"
    assert "compare" not in state["stage_history"]


def test_compare_is_legal_only_from_explore_or_compare_and_records_the_stage(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _run(capsys, ["--campaign", str(campaign), "--record", str(evidence), "--json"])
    _run(capsys, ["--campaign", str(campaign), "--goal", "prefill_tok_s >= 1200", "--json"])

    rc, payload, _ = _run(capsys, ["--campaign", str(campaign), "--compare", "--json"])
    assert rc != 0
    assert "explore" in payload["error"].lower()

    for stage in ("discover", "explore"):
        _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])
    rc, compared, _ = _run(capsys, ["--campaign", str(campaign), "--compare", "--json"])
    assert rc == 0
    assert compared["stage"] == "compare"
    assert compared["stage_history"] == ["observe", "discover", "explore", "compare"]

    rc, compared_again, _ = _run(capsys, ["--campaign", str(campaign), "--compare", "--json"])
    assert rc == 0
    assert compared_again["stage_history"].count("compare") == 1


def test_promote_requires_explain_or_promote_stage(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _, recorded, _ = _run(capsys, ["--campaign", str(campaign), "--record", str(evidence), "--json"])
    evidence_id = recorded["frontier"][0]

    rc, payload, _ = _run(capsys, ["--campaign", str(campaign), "--promote", evidence_id, "--json"])
    assert rc != 0
    assert "explain" in payload["error"].lower()

    for stage in ("discover", "explore", "explain"):
        _run(capsys, ["--campaign", str(campaign), "--advance", stage, "--json"])
    rc, promoted, _ = _run(capsys, ["--campaign", str(campaign), "--promote", evidence_id, "--json"])
    assert rc == 0
    assert promoted["stage"] == "promote"
    assert promoted["promotion"] == evidence_id


def test_combined_record_and_invalid_goal_is_atomic(tmp_path, capsys):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _run(capsys, ["--campaign", str(campaign), "--json"])

    rc, payload, _ = _run(
        capsys,
        ["--campaign", str(campaign), "--record", str(evidence), "--goal", "not a goal", "--json"],
    )
    assert rc != 0
    assert "goal" in payload["error"].lower()
    _, state, _ = _run(capsys, ["--campaign", str(campaign), "--json"])
    assert state["evidence"] == []
    assert state["references"] == []
    evidence_dir = campaign.parent / f".{campaign.stem}.evidence"
    assert not evidence_dir.exists() or list(evidence_dir.iterdir()) == []


def test_failed_comparison_keeps_a_valid_new_reference(tmp_path, capsys, monkeypatch):
    campaign = tmp_path / "campaign.json"
    evidence = tmp_path / "measurement.json"
    references = tmp_path / "references"
    _write(campaign, _campaign())
    _write(evidence, _evidence())
    _write(references / "upstream-llama.json", _reference())
    monkeypatch.setenv("AUTOLUCE_REFERENCE_DIR", str(references))
    _run(capsys, ["--campaign", str(campaign), "--record", str(evidence), "--json"])

    rc, payload, _ = _run(
        capsys,
        ["--campaign", str(campaign), "--against", "upstream-llama", "--compare", "--json"],
    )
    assert rc != 0
    assert "explore" in payload["error"].lower()
    _, state, _ = _run(capsys, ["--campaign", str(campaign), "--json"])
    assert [item["name"] for item in state["references"]] == ["upstream-llama"]
    assert state["comparisons"] == []


def test_readme_goal_happy_path_matches_the_campaign_example():
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    example = json.loads((root / "examples" / "research-campaign.json").read_text())
    metric = example["objective"]["metric"]
    assert f"--goal '{metric} >= 1500' --compare" in readme
    assert "--advance explore" in readme
