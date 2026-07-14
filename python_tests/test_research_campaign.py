"""Contract tests for the machine-aware research campaign vertical slice.

These tests intentionally describe domain behavior, not CLI plumbing.  The same
objects are expected to back both the friendly CLI and its deterministic JSON form.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from autoluce.contracts import ResearchContract
from autoluce.research import (
    Campaign,
    CampaignEvidence,
    CampaignStore,
    CompatibilityError,
    ParetoArchive,
    Reference,
    compare_state,
    migrate_v1_contract,
    parse_goal_reference,
)
from autoluce.research_contract import content_id


SYSTEM = {
    "machine": "machine-a",
    "model": "qwen3.6-27b",
    "model_fingerprint": "model-a",
    "runtime": "lucebox",
    "hardware": "rtx-3090",
    "backend": "cuda",
    "quantization": "iq4_xs",
    "environment": "env-a",
}
WORKLOAD = {
    "contexts": [8192, 32768],
    "batch_shape": {"batch": 1, "ubatch": 1},
    "mode": "prefill",
    "prompts": ["Explain why the sky is blue."],
}
OBJECTIVE = {"metric": "prefill_tok_s", "direction": "maximize"}
CONSTRAINTS = {"correctness": "exact", "quality": {"kl_mean_max": 0.01}}


def campaign(**overrides) -> Campaign:
    values = {
        "name": "prefill-research",
        "system": SYSTEM,
        "workload": WORKLOAD,
        "objective": OBJECTIVE,
        "constraints": CONSTRAINTS,
    }
    values.update(overrides)
    return Campaign(**values)


@pytest.mark.parametrize("missing", ["system", "workload", "objective", "constraints"])
def test_campaign_requires_every_research_dimension(missing):
    values = {
        "name": "incomplete",
        "system": SYSTEM,
        "workload": WORKLOAD,
        "objective": OBJECTIVE,
        "constraints": CONSTRAINTS,
    }
    values[missing] = None

    with pytest.raises(ValueError, match=missing):
        Campaign(**values)


def test_reference_is_optional_and_all_supported_reference_concepts_are_explicit():
    assert campaign().reference is None

    references = [
        Reference(kind="accepted_baseline"),
        Reference(kind="executable", locator="upstream-llama"),
        Reference(kind="candidate", locator="refs/heads/feature-x"),
        Reference(kind="result_bundle", locator="results/frozen/run-17"),
        Reference(kind="measurement", value=1475.0, metric="prefill_tok_s", provenance="published/manual"),
        parse_goal_reference("prefill_tok_s >= 1500"),
    ]

    assert [reference.kind for reference in references] == [
        "accepted_baseline",
        "executable",
        "candidate",
        "result_bundle",
        "measurement",
        "goal",
    ]
    goal = references[-1]
    assert (goal.metric, goal.operator, goal.value) == ("prefill_tok_s", ">=", 1500.0)


def test_attaching_reference_preserves_previously_collected_immutable_evidence():
    observed = CampaignEvidence.create(
        campaign_id=campaign().campaign_id,
        system=SYSTEM,
        workload=WORKLOAD,
        metrics={"prefill_tok_s": 1400.0, "peak_mem_gib": 18.0},
        gates={"correctness": True, "quality": True},
        artifact_hash="sha256:artifact-a",
    )
    original = campaign(evidence=(observed,))
    evidence_id = original.evidence[0].evidence_id

    compared = original.attach_reference(Reference(kind="executable", locator="upstream-llama"))

    assert original.reference is None
    assert compared.reference == Reference(kind="executable", locator="upstream-llama")
    assert compared.evidence == original.evidence
    assert compared.evidence[0].evidence_id == evidence_id


def evidence(**system_overrides) -> CampaignEvidence:
    return CampaignEvidence.create(
        campaign_id="campaign-a",
        system={**SYSTEM, **system_overrides},
        workload=WORKLOAD,
        metrics={"prefill_tok_s": 1400.0},
        gates={"correctness": True, "quality": True},
        artifact_hash="sha256:artifact-a",
    )


@pytest.mark.parametrize(
    ("dimension", "candidate"),
    [
        ("machine", evidence(machine="machine-b")),
        ("model", evidence(model="different-model")),
        ("runtime", evidence(runtime="upstream-llama")),
        ("backend", evidence(backend="vulkan")),
        ("quantization", evidence(quantization="q4_k_m")),
        ("environment", evidence(environment="env-b")),
        (
            "workload",
            replace(evidence(), workload={**WORKLOAD, "contexts": [8192]}),
        ),
    ],
)
def test_comparison_rejects_every_incompatible_evidence_dimension(dimension, candidate):
    with pytest.raises(CompatibilityError, match=dimension):
        evidence().compare(candidate)


def test_quality_constrained_pareto_archive_keeps_multiple_elites_and_all_evidence():
    archive = ParetoArchive(objectives={"prefill_tok_s": "maximize", "peak_mem_gib": "minimize"})
    fast = CampaignEvidence.create(
        campaign_id="campaign-a", system=SYSTEM, workload=WORKLOAD,
        metrics={"prefill_tok_s": 1600.0, "peak_mem_gib": 22.0},
        gates={"correctness": True, "quality": True}, artifact_hash="sha256:fast",
    )
    lean = CampaignEvidence.create(
        campaign_id="campaign-a", system=SYSTEM, workload=WORKLOAD,
        metrics={"prefill_tok_s": 1450.0, "peak_mem_gib": 17.0},
        gates={"correctness": True, "quality": True}, artifact_hash="sha256:lean",
    )
    dominated = CampaignEvidence.create(
        campaign_id="campaign-a", system=SYSTEM, workload=WORKLOAD,
        metrics={"prefill_tok_s": 1400.0, "peak_mem_gib": 20.0},
        gates={"correctness": True, "quality": True}, artifact_hash="sha256:dominated",
    )
    invalid = CampaignEvidence.create(
        campaign_id="campaign-a", system=SYSTEM, workload=WORKLOAD,
        metrics={"prefill_tok_s": 2000.0, "peak_mem_gib": 16.0},
        gates={"correctness": True, "quality": False}, artifact_hash="sha256:invalid",
    )

    for item in (fast, lean, dominated, invalid):
        archive.add(item)

    assert {item.evidence_id for item in archive.frontier} == {fast.evidence_id, lean.evidence_id}
    assert {item.evidence_id for item in archive.evidence} == {
        fast.evidence_id,
        lean.evidence_id,
        dominated.evidence_id,
        invalid.evidence_id,
    }


def test_v1_research_contract_migrates_explicitly_without_inventing_a_reference():
    legacy = ResearchContract(
        target="strix",
        machine_fingerprint="machine-a",
        model="deepseek-v4-flash",
        model_fingerprint="model-a",
        workload="interactive_single_user",
        contexts=[8192, 32768],
        primary_objective="interactive_decode",
        backends=["hip"],
        primary_backend="hip",
    )

    migrated = migrate_v1_contract(legacy.to_dict())

    assert migrated.schema_version == 2
    assert migrated.system["machine"] == "machine-a"
    assert migrated.system["model"] == "deepseek-v4-flash"
    assert migrated.system["backend"] == "hip"
    assert migrated.workload["contexts"] == [8192, 32768]
    assert migrated.objective == {"metric": "decode_tok_s", "direction": "maximize"}
    assert migrated.reference is None


def test_contract_migration_rejects_unknown_future_versions():
    with pytest.raises(ValueError, match="unsupported.*schema.*3"):
        migrate_v1_contract({"schema_version": 3})


def test_v1_operational_constraints_migrate_to_enforceable_normalized_gates():
    legacy = ResearchContract(
        target="strix",
        machine_fingerprint="machine-a",
        model="deepseek-v4-flash",
        model_fingerprint="model-a",
        backends=["hip"],
        primary_backend="hip",
        host_headroom_gib=16.0,
        baseline_fraction_min=0.97,
        power_mode="maximum_performance",
    )

    migrated = migrate_v1_contract(legacy.to_dict())

    assert migrated.constraints["gates"] == [
        {"metric": "host_headroom_gib", "operator": ">=", "value": 16.0},
        {
            "metric": "accepted_baseline_fraction",
            "operator": ">=",
            "value": 0.97,
            "verification": "comparison_required",
        },
        {"metric": "power_mode", "operator": "==", "value": "maximum_performance"},
    ]
    assert migrated.constraints["quality"] == {
        "kind": "kl",
        "mean_max": 0.01,
        "max_max": 0.1,
    }


def test_resolved_migrated_campaign_defers_baseline_gate_without_losing_frontier_evidence(tmp_path):
    legacy = ResearchContract(
        target="strix", machine_fingerprint="machine-a", model="deepseek-v4-flash",
        model_fingerprint="model-a", backends=["hip"], primary_backend="hip",
        host_headroom_gib=16.0, baseline_fraction_min=0.97,
    )
    migrated = migrate_v1_contract(legacy.to_dict())
    resolved = replace(migrated, system={
        **migrated.system,
        "runtime": "lucebox:dflash-server-http",
        "hardware": "strix-halo",
        "quantization": "IQ2XXS",
        "environment": "env-a",
    })
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(resolved.to_dict()))
    store = CampaignStore(path)
    state = store.load()

    recorded = store.record(state, {
        "metrics": {"decode_tok_s": 20.0, "host_headroom_gib": 20.0},
        "quality": {
            "correctness": {"kind": "exact", "passed": True},
            "kl": {"kind": "kl", "mean": 0.001, "max": 0.01},
        },
        "artifact_hash": "sha256:resolved",
        "provenance": {"power_mode": "maximum_performance"},
    })

    assert recorded.feasible is True
    assert recorded.provenance["deferred_constraints"] == ["accepted_baseline_fraction"]
    assert state["frontier"] == [recorded.evidence_id]


def _reference_entry(reference: Reference, **extra):
    value = {**reference.to_dict(), **extra}
    return {**value, "reference_id": content_id("reference", value)}


def _reference_compatibility(runtime="lucebox"):
    return {
        "machine": SYSTEM["machine"],
        "model": SYSTEM["model"],
        "model_fingerprint": SYSTEM["model_fingerprint"],
        "runtime": runtime,
        "hardware": SYSTEM["hardware"],
        "backend": SYSTEM["backend"],
        "quantization": SYSTEM["quantization"],
        "environment": SYSTEM["environment"],
        "workload": campaign().name,
        "workload_fingerprint": content_id("workload", WORKLOAD),
    }


def _comparison_state(reference, baseline=None, candidate=None):
    baseline = baseline or evidence()
    candidate = candidate or CampaignEvidence.create(
        campaign_id="campaign-a",
        system=SYSTEM,
        workload=WORKLOAD,
        metrics={"prefill_tok_s": 1500.0},
        gates={"correctness": True, "quality": True},
        artifact_hash="sha256:candidate",
    )
    return {
        "campaign": campaign().to_dict(include_evidence=False),
        "evidence": [baseline.to_dict(), candidate.to_dict()],
        "frontier": [baseline.evidence_id, candidate.evidence_id],
        "references": [reference],
        "comparisons": [],
        "promotion": baseline.evidence_id,
    }


@pytest.mark.parametrize("kind", ["measurement", "candidate"])
def test_non_runtime_references_reject_runtime_mismatch(kind):
    reference = Reference(
        kind=kind,
        locator="refs/heads/candidate" if kind == "candidate" else None,
        metric="prefill_tok_s" if kind == "measurement" else None,
        value=1425.0 if kind == "measurement" else None,
        compatibility=_reference_compatibility(runtime="upstream-llama"),
    )
    entry = _reference_entry(reference, measurement={"prefill_tok_s": 1425.0})

    with pytest.raises(CompatibilityError, match="runtime"):
        compare_state(_comparison_state(entry))


def test_runtime_reference_explicitly_allows_runtime_identity_to_vary():
    reference = Reference(
        kind="runtime",
        locator="upstream-llama",
        compatibility=_reference_compatibility(runtime="upstream-llama"),
    )
    entry = _reference_entry(
        reference,
        measurement={"prefill_tok_s": 1425.0},
        allowed_system_variations=["runtime"],
    )

    result = compare_state(_comparison_state(entry))

    assert result["compatible"] is True
    assert result["kind"] == "runtime"
    assert result["delta"] == 75.0


def test_fully_identified_v2_evidence_compares_against_accepted_baseline():
    state = _comparison_state(_reference_entry(Reference(kind="accepted_baseline")))

    result = compare_state(state)

    assert result["compatible"] is True
    assert result["kind"] == "accepted_baseline"
    assert result["reference_evidence_id"] == state["promotion"]
    assert result["deltas"]["prefill_tok_s"] == 100.0
    assert result["accepted_baseline_fraction"] == pytest.approx(1500.0 / 1400.0)


def test_migrated_unknown_identities_cannot_be_recorded_until_resolved(tmp_path):
    legacy = ResearchContract(
        target="strix",
        machine_fingerprint="machine-a",
        model="deepseek-v4-flash",
        model_fingerprint="model-a",
        backends=["hip"],
        primary_backend="hip",
    )
    migrated = migrate_v1_contract(legacy.to_dict())
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(migrated.to_dict()))
    store = CampaignStore(path)
    state = store.load()

    with pytest.raises(ValueError, match="unresolved.*runtime.*hardware.*quantization"):
        store.record(state, {
            "metrics": {"decode_tok_s": 20.0, "host_headroom_gib": 20.0},
            "quality": {
                "correctness": {"kind": "exact", "passed": True},
                "kl": {"kind": "kl", "mean": 0.001, "max": 0.01},
            },
            "artifact_hash": "sha256:legacy-candidate",
            "provenance": {"power_mode": "maximum_performance"},
        })


@pytest.mark.parametrize(
    ("quality", "violation"),
    [
        (
            {
                "correctness": {"kind": "exact", "passed": False},
                "kl": {"kind": "kl", "mean": 0.001, "max": 0.01},
            },
            "correctness",
        ),
        (
            {
                "correctness": {"kind": "exact", "passed": True},
                "kl": {"kind": "kl", "mean": 0.02, "max": 0.08},
            },
            "kl.mean",
        ),
        (
            {
                "correctness": {"kind": "exact", "passed": True},
                "kl": {"kind": "kl", "mean": 0.005, "max": 0.2},
            },
            "kl.max",
        ),
    ],
)
def test_typed_exact_and_kl_quality_evidence_enforces_campaign_thresholds(tmp_path, quality, violation):
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(campaign().to_dict()))
    store = CampaignStore(path)
    state = store.load()

    recorded = store.record(state, {
        "metrics": {"prefill_tok_s": 1500.0},
        "quality": quality,
        "artifact_hash": f"sha256:{violation}",
    })

    assert recorded.feasible is False
    violations = recorded.provenance["campaign_constraint_violations"]
    assert any(violation in item for item in violations)


def test_typed_exact_and_kl_quality_evidence_can_pass_all_thresholds(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(campaign().to_dict()))
    store = CampaignStore(path)
    state = store.load()

    recorded = store.record(state, {
        "metrics": {"prefill_tok_s": 1500.0},
        "quality": {
            "correctness": {"kind": "exact", "passed": True},
            "kl": {"kind": "kl", "mean": 0.005, "max": 0.05},
        },
        "artifact_hash": "sha256:quality-pass",
    })

    assert recorded.feasible is True
    assert "campaign_constraint_violations" not in recorded.provenance


def test_harness_exact_pass_is_normalized_to_typed_campaign_correctness(tmp_path):
    exact_campaign = campaign(constraints={"correctness": "exact"})
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(exact_campaign.to_dict()))
    store = CampaignStore(path)
    state = store.load()

    raw = {
        "prefill_tok_s": 1500.0,
        "correctness": "pass",
        "source_evidence": {"binary_sha256": "sha256:harness"},
        "benchmarks": [{
            "context_metrics": [{"context_depth": 8192, "prefill_tok_s_samples": [1498, 1500, 1502]}],
        }],
    }
    recorded = store.record(state, raw)

    assert recorded.gates["correctness"] is True
    assert recorded.feasible is True
    assert state["frontier"] == [recorded.evidence_id]
    assert "campaign_constraint_violations" not in recorded.provenance
    measurement_id = recorded.provenance["measurement_bundle_id"]
    measurement_path = store.evidence_dir / f"{measurement_id}.json"
    assert __import__("json").loads(measurement_path.read_text()) == raw


def test_diagnostic_workload_cannot_enter_the_frontier(tmp_path):
    diagnostic = campaign(workload={**WORKLOAD, "frontier_eligible": False})
    path = tmp_path / "campaign.json"
    path.write_text(__import__("json").dumps(diagnostic.to_dict()))
    store = CampaignStore(path)
    state = store.load()

    recorded = store.record(state, {
        "metrics": {"prefill_tok_s": 1500.0},
        "quality": {
            "correctness": {"kind": "exact", "passed": True},
            "kl": {"kind": "kl", "mean": 0.005, "max": 0.05},
        },
        "artifact_hash": "sha256:diagnostic",
    })

    assert recorded.gates["frontier_eligible"] is False
    assert recorded.feasible is False
    assert state["frontier"] == []
