"""Machine-aware research campaigns and the ``autoluce research`` entry point.

This module is intentionally an orchestrator.  Contract/reference behavior lives in
``research_contract``; immutable measurements and compatibility in
``research_evidence``; frontier policy in ``research_archive``.  Re-exports keep a
small stable API for callers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from autoluce.research_archive import ParetoArchive
from autoluce.bench.uncertainty import is_significant_improvement
from autoluce.research_contract import (
    LIFECYCLE,
    Campaign,
    Reference,
    content_id,
    migrate_v1_contract,
    parse_against_reference,
    parse_goal_reference,
    validate_lifecycle_transition,
)
from autoluce.research_evidence import CampaignEvidence, CompatibilityError
from autoluce.prefill_plan import promotion_bundle_violations


DEFAULT_CAMPAIGN = Path(".autoluce/research/campaign.json")


def _campaign_constraint_violations(
    metrics: Mapping[str, float],
    quality: Mapping[str, Any],
    constraints: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
) -> list[str]:
    attributes = attributes or {}
    violations: list[str] = []
    for metric, rule in constraints.items():
        if metric == "correctness":
            evidence = quality.get("correctness")
            if isinstance(evidence, Mapping):
                passed = evidence.get("passed") is True
                kind = evidence.get("kind")
            else:
                passed = str(evidence).lower() == "pass"
                kind = quality.get("correctness_oracle")
            if not passed or (rule and str(rule) != str(kind)):
                violations.append("correctness")
            continue
        if metric == "quality" and isinstance(rule, Mapping):
            kind = str(rule.get("kind", "kl" if any("kl" in key for key in rule) else ""))
            evidence = quality.get(kind) if kind else None
            if not isinstance(evidence, Mapping) or evidence.get("kind", kind) != kind:
                violations.append(f"quality.{kind or 'evidence'}: not measured")
                continue
            mean_bound = rule.get("mean_max", rule.get("kl_mean_max"))
            max_bound = rule.get("max_max", rule.get("kl_max"))
            if max_bound is None and kind == "kl" and mean_bound is not None:
                max_bound = 10 * float(mean_bound)
            if mean_bound is not None:
                mean = evidence.get("mean", evidence.get("mean_kl_divergence"))
                if mean is None or float(mean) > float(mean_bound):
                    violations.append(f"{kind}.mean")
            if max_bound is not None:
                maximum = evidence.get("max", evidence.get("max_kl_divergence"))
                if maximum is None or float(maximum) > float(max_bound):
                    violations.append(f"{kind}.max")
            continue
        if not isinstance(rule, Mapping):
            continue
        if metric not in metrics:
            violations.append(f"{metric}: not measured")
            continue
        value = float(metrics[metric])
        if "max" in rule and value > float(rule["max"]):
            violations.append(f"{metric}: {value} > max {rule['max']}")
        if "min" in rule and value < float(rule["min"]):
            violations.append(f"{metric}: {value} < min {rule['min']}")
        if "min_frac_of_baseline" in rule and not (
            quality.get("constraints") is True
            or str(quality.get("constraints", "")).lower() == "pass"
        ):
            violations.append(f"{metric}: baseline-relative constraint was not verified")
        unknown_forms = set(rule) - {"min", "max", "min_frac_of_baseline"}
        if unknown_forms:
            violations.append(f"{metric}: unsupported constraint forms {sorted(unknown_forms)}")
    for rule in constraints.get("gates", []):
        if not isinstance(rule, Mapping):
            continue
        metric, operator, target = rule.get("metric"), rule.get("operator"), rule.get("value")
        if metric == "correctness":
            value = quality.get("correctness")
        elif metric == "host_headroom_gib":
            value = metrics.get("host_headroom_gib", metrics.get("min_mem_available_GiB"))
        elif metric == "power_mode":
            value = attributes.get("power_mode")
        else:
            value = metrics.get(str(metric))
        if rule.get("verification") == "comparison_required" and value is None:
            continue
        if operator == "==":
            passed = value == target
        elif operator == ">=":
            passed = value is not None and float(value) >= float(target)
        elif operator == "<=":
            passed = value is not None and float(value) <= float(target)
        else:
            passed = False
        if not passed:
            violations.append(f"{metric} {operator} {target}")
    return violations


def _quality_gate_results(quality: Mapping[str, Any]) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for name, value in quality.items():
        if isinstance(value, Mapping):
            if "passed" in value:
                gates[name] = value.get("passed") is True
            elif value.get("kind") == "kl":
                gates[name] = value.get("mean") is not None and value.get("max") is not None
            else:
                gates[name] = False
        else:
            gates[name] = value is True or str(value).lower() == "pass"
    return gates


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _legacy_campaign(value: Mapping[str, Any]) -> Campaign:
    """Read the short-lived campaign-v1 shape without confusing it with v1 execution contracts."""
    system = dict(value.get("system_under_test", {}))
    hardware = str(system.get("hardware", "unknown:campaign-v1"))
    system.setdefault("machine", hardware)
    system.setdefault("model_fingerprint", "unknown:campaign-v1")
    system.setdefault("environment", "unknown:campaign-v1")
    constraints = value.get("constraints", {})
    if isinstance(constraints, list):
        constraints = {"gates": constraints}
    return Campaign(
        name=str(value.get("name", "research")),
        system=system,
        workload=dict(value.get("workload", {})),
        objective=dict(value.get("objective", {})),
        constraints=dict(constraints),
    )


def load_campaign(path: Path) -> Campaign:
    value = json.loads(path.read_text())
    if "system_under_test" in value:
        return _legacy_campaign(value)
    if int(value.get("schema_version", 1)) == 1 and "machine_fingerprint" in value:
        return migrate_v1_contract(value)
    return Campaign.from_dict(value)


class CampaignStore:
    """Filesystem adapter for one campaign and its immutable evidence files."""

    def __init__(self, campaign_path: Path) -> None:
        self.campaign_path = campaign_path
        self.state_path = campaign_path.with_suffix(campaign_path.suffix + ".state.json")
        self.evidence_dir = campaign_path.parent / f".{campaign_path.stem}.evidence"

    def load(self) -> dict[str, Any]:
        campaign = load_campaign(self.campaign_path)
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("campaign_id") != campaign.campaign_id:
                raise ValueError("campaign measurement scope changed; start a new campaign archive")
            return state
        return {
            "schema_version": 1,
            "campaign_id": campaign.campaign_id,
            "campaign": campaign.to_dict(include_evidence=False),
            "stage": "observe",
            "stage_history": ["observe"],
            "references": [],
            "evidence": [],
            "frontier": [],
            "comparisons": [],
            "promotion": None,
        }

    def save(self, state: Mapping[str, Any]) -> None:
        _atomic_write(self.state_path, state)

    def record(self, state: dict[str, Any], raw: Mapping[str, Any]) -> CampaignEvidence:
        campaign = Campaign.from_dict(state["campaign"])
        system = dict(campaign.system)
        unresolved = [
            name for name in ("runtime", "hardware", "quantization")
            if str(system.get(name, "")).startswith("unknown:")
        ]
        if unresolved:
            raise ValueError(
                "unresolved system identities: " + ", ".join(unresolved)
                + "; create a fully observed v2 campaign before recording evidence"
            )
        system.setdefault("machine", system.get("hardware"))
        provenance = dict(raw.get("provenance", raw.get("source_evidence", {})))
        measurement_bundle_id = content_id("measurement", raw)
        if raw.get("experiment"):
            provenance["experiment"] = raw["experiment"]
        if str(system.get("environment", "")).startswith("unknown:"):
            system["environment"] = content_id("environment", provenance)
        # Archive identity is interpretation metadata, not part of the measured
        # execution environment used for compatibility checks.
        provenance["measurement_bundle_id"] = measurement_bundle_id
        quality = dict(raw.get("quality", {}))
        if "correctness" in raw:
            raw_correctness = raw["correctness"]
            if "correctness" not in quality:
                correctness_kind = campaign.constraints.get("correctness")
                passed = raw_correctness is True or str(raw_correctness).lower() == "pass"
                if isinstance(correctness_kind, str) and correctness_kind:
                    quality["correctness"] = {
                        "kind": correctness_kind,
                        "passed": passed,
                    }
                else:
                    quality["correctness"] = raw_correctness
            quality.setdefault("quality", raw["correctness"])
        benchmark_gates = [
            not item.get("constraint_violations")
            for item in raw.get("benchmarks", [])
        ]
        if benchmark_gates:
            quality.setdefault("constraints", all(benchmark_gates))
        gates = _quality_gate_results(quality)
        if not gates:
            gates = {"quality": False}
        if campaign.workload.get("frontier_eligible") is False:
            gates["frontier_eligible"] = False
            provenance["frontier_eligible"] = False
        artifact_hash = str(
            raw.get("artifact_hash")
            or provenance.get("binary_sha256")
            or provenance.get("product_digest")
            or content_id("artifact", provenance)
        )
        raw_metrics = raw.get("metrics")
        if raw_metrics is None:
            raw_metrics = {
                name: value for name, value in raw.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        metrics = {name: float(value) for name, value in raw_metrics.items()}
        if not metrics:
            raise ValueError("measurement did not contain numeric metrics")
        constraint_violations = _campaign_constraint_violations(
            metrics, quality, campaign.constraints, provenance,
        )
        deferred_constraints = [
            str(rule.get("metric"))
            for rule in campaign.constraints.get("gates", [])
            if isinstance(rule, Mapping) and rule.get("verification") == "comparison_required"
        ]
        if deferred_constraints:
            provenance["deferred_constraints"] = deferred_constraints
        if constraint_violations:
            gates["campaign_constraints"] = False
            provenance["campaign_constraint_violations"] = constraint_violations
        evidence_profile = campaign.constraints.get("evidence_profile")
        if evidence_profile == "normal_kv_prefill_v1":
            promotion_violations = promotion_bundle_violations(raw, campaign)
            gates["promotion_evidence"] = not promotion_violations
            if promotion_violations:
                provenance["promotion_evidence_violations"] = promotion_violations
        elif evidence_profile is not None:
            gates["promotion_evidence"] = False
            provenance["promotion_evidence_violations"] = [
                f"unsupported evidence profile '{evidence_profile}'"
            ]
        uncertainty = {
            name.removesuffix("_stddev"): value
            for name, value in metrics.items() if name.endswith("_stddev")
        }
        evidence = CampaignEvidence.create(
            campaign_id=campaign.campaign_id,
            system=system,
            workload=campaign.workload,
            metrics=metrics,
            gates=gates,
            artifact_hash=artifact_hash,
            uncertainty=uncertainty,
            provenance=provenance,
        )
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        measurement_path = self.evidence_dir / f"{measurement_bundle_id}.json"
        if not measurement_path.exists():
            with measurement_path.open("x") as stream:
                json.dump(raw, stream, indent=2, sort_keys=True)
                stream.write("\n")
        evidence_path = self.evidence_dir / f"{evidence.evidence_id}.json"
        if not evidence_path.exists():
            # Content address plus exclusive creation makes evidence immutable.
            with evidence_path.open("x") as stream:
                json.dump(evidence.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
        if not any(item["evidence_id"] == evidence.evidence_id for item in state["evidence"]):
            state["evidence"].append(evidence.to_dict())
        self._refresh_frontier(state)
        return evidence

    @staticmethod
    def _refresh_frontier(state: dict[str, Any]) -> None:
        campaign = Campaign.from_dict(state["campaign"])
        objectives = {campaign.objective["metric"]: campaign.objective["direction"]}
        archive = ParetoArchive(objectives)
        for raw in state["evidence"]:
            archive.add(CampaignEvidence.from_dict(raw))
        state["frontier"] = [item.evidence_id for item in archive.frontier]


def _append_reference(state: dict[str, Any], reference: Mapping[str, Any]) -> None:
    identity = content_id("reference", reference)
    entry = {**dict(reference), "reference_id": identity}
    if not any(item.get("reference_id") == identity for item in state["references"]):
        state["references"].append(entry)


def _load_named_reference(name: str) -> dict[str, Any]:
    root = Path(os.environ.get("AUTOLUCE_REFERENCE_DIR", "references"))
    path = root / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    reference = parse_against_reference(name).to_dict()
    if reference["kind"] == "result_bundle":
        bundle_path = Path(reference["locator"])
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text())
            try:
                evidence = CampaignEvidence.from_dict(bundle)
            except (TypeError, ValueError):
                # Legacy bundles stay attachable for provenance, but without a
                # compatibility descriptor they remain deliberately uncomparable.
                reference["legacy_bundle"] = True
            else:
                reference["evidence"] = evidence.to_dict()
    return reference


def _reference_mismatches(
    campaign: Campaign,
    candidate: CampaignEvidence,
    reference: Mapping[str, Any],
) -> list[str]:
    compatibility = reference.get("compatibility")
    if not compatibility:
        # A named executable/candidate is attachable for planning, but comparison
        # waits for measured compatibility metadata.
        return ["reference compatibility metadata"]
    allowed = set(reference.get("allowed_system_variations", []))
    if reference.get("kind") in {"runtime", "executable"}:
        allowed.add("runtime")
    expected = {
        "machine": candidate.system.get("machine"),
        "model": candidate.system.get("model"),
        "hardware": candidate.system.get("hardware"),
        "backend": candidate.system.get("backend"),
        "quantization": candidate.system.get("quantization"),
        "environment": candidate.system.get("environment"),
        "workload": campaign.name,
        "workload_fingerprint": content_id("workload", candidate.workload),
    }
    if candidate.system.get("model_fingerprint") is not None:
        expected["model_fingerprint"] = candidate.system["model_fingerprint"]
    if "runtime" not in allowed:
        expected["runtime"] = candidate.system.get("runtime")
    return [
        name for name, value in expected.items()
        if name not in allowed
        and (str(value).startswith("unknown:") or compatibility.get(name) != value)
    ]


def _goal_satisfied(value: float, operator: str, target: float) -> bool:
    return {
        ">=": value >= target,
        "<=": value <= target,
        ">": value > target,
        "<": value < target,
        "==": value == target,
    }[operator]


def compare_state(state: dict[str, Any]) -> dict[str, Any]:
    if not state["evidence"]:
        raise ValueError("compare requires campaign evidence")
    if not state["references"]:
        raise ValueError("compare requires a performance reference")
    campaign = Campaign.from_dict(state["campaign"])
    candidate = CampaignEvidence.from_dict(state["evidence"][-1])
    reference = state["references"][-1]
    if reference["kind"] == "goal":
        metric = str(reference["metric"])
        if metric not in candidate.metrics:
            raise ValueError(f"goal metric '{metric}' was not measured")
        value, target = float(candidate.metrics[metric]), float(reference["value"])
        sigma = float(candidate.metrics.get(f"{metric}_stddev", 0.0))
        conservative_value = value
        if reference["operator"] in {">", ">="}:
            conservative_value -= sigma
        elif reference["operator"] in {"<", "<="}:
            conservative_value += sigma
        result = {
            "kind": "goal",
            "reference_id": reference["reference_id"],
            "candidate_evidence_id": candidate.evidence_id,
            "metric": metric,
            "value": value,
            "target": target,
            "operator": reference["operator"],
            "uncertainty": sigma,
            "matched": _goal_satisfied(conservative_value, str(reference["operator"]), target),
        }
    elif reference["kind"] == "result_bundle" and reference.get("evidence"):
        reference_evidence = CampaignEvidence.from_dict(reference["evidence"])
        allowed = frozenset(reference.get("allowed_system_variations", []))
        result = reference_evidence.compare(candidate, allowed_system_variations=allowed)
        result.update({"kind": "result_bundle", "reference_id": reference["reference_id"]})
    elif reference["kind"] == "accepted_baseline":
        baseline_id = state.get("promotion") or next(
            (item for item in state["frontier"] if item != candidate.evidence_id), None
        )
        if not baseline_id:
            raise ValueError("accepted baseline reference requires an earlier promoted or frontier result")
        baseline = CampaignEvidence.from_dict(next(
            item for item in state["evidence"] if item["evidence_id"] == baseline_id
        ))
        result = baseline.compare(candidate)
        metric = str(campaign.objective["metric"])
        if metric in baseline.metrics and metric in candidate.metrics:
            baseline_value = float(baseline.metrics[metric])
            if baseline_value:
                result["accepted_baseline_fraction"] = float(candidate.metrics[metric]) / baseline_value
        result.update({"kind": "accepted_baseline", "reference_id": reference["reference_id"]})
    else:
        mismatches = _reference_mismatches(campaign, candidate, reference)
        if mismatches:
            raise CompatibilityError(mismatches)
        measurement = dict(reference.get("measurement", {}))
        if reference["kind"] == "measurement" and reference.get("metric") is not None:
            measurement[str(reference["metric"])] = reference.get("value")
        metric = str(campaign.objective["metric"])
        if metric not in measurement:
            raise ValueError(f"reference did not measure '{metric}'")
        candidate_value, reference_value = float(candidate.metrics[metric]), float(measurement[metric])
        candidate_sigma = float(candidate.metrics.get(f"{metric}_stddev", 0.0))
        reference_sigma = float(measurement.get(f"{metric}_stddev", 0.0))
        direction = str(campaign.objective["direction"])
        signed_delta = candidate_value - reference_value
        significant = is_significant_improvement(
            candidate_value if direction == "maximize" else -candidate_value,
            candidate_sigma,
            reference_value if direction == "maximize" else -reference_value,
            reference_sigma,
        )
        result = {
            "kind": reference["kind"],
            "reference_id": reference["reference_id"],
            "candidate_evidence_id": candidate.evidence_id,
            "metric": metric,
            "delta": signed_delta,
            "significant_improvement": significant,
            "compatible": True,
        }
    state["comparisons"].append(result)
    return result


def _promotion_constraint_violations(state: Mapping[str, Any], evidence_id: str) -> list[str]:
    campaign = Campaign.from_dict(state["campaign"])
    violations: list[str] = []
    for rule in campaign.constraints.get("gates", []):
        if not isinstance(rule, Mapping) or rule.get("verification") != "comparison_required":
            continue
        metric = str(rule.get("metric"))
        # The first accepted result establishes the local baseline. Later results
        # must resolve baseline-relative gates through an explicit comparison.
        if metric == "accepted_baseline_fraction" and state.get("promotion") is None:
            continue
        comparison = next(
            (
                item for item in reversed(state["comparisons"])
                if item.get("candidate_evidence_id") == evidence_id
                and item.get("kind") == "accepted_baseline"
            ),
            None,
        )
        if comparison is None:
            violations.append(f"{metric}: accepted-baseline comparison required")
            continue
        value = comparison.get(metric)
        target = float(rule["value"])
        if value is None or (rule.get("operator") == ">=" and float(value) < target):
            violations.append(f"{metric}: {value} does not satisfy {rule.get('operator')} {target}")
    return violations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one machine-aware AutoLuce research campaign")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--record", type=Path, help="archive a measurement JSON without requiring a reference")
    parser.add_argument("--against", help="attach an accepted baseline, executable, candidate, or bundle reference")
    parser.add_argument("--goal", help="attach an absolute goal such as 'prefill_tok_s >= 1500'")
    parser.add_argument("--compare", action="store_true", help="interpret the newest evidence against the newest reference")
    parser.add_argument("--advance", choices=LIFECYCLE, help="advance the campaign lifecycle")
    parser.add_argument("--promote", metavar="EVIDENCE_ID", help="explicitly promote a frontier evidence record")
    parser.add_argument("--json", action="store_true", help="emit one deterministic JSON document")
    return parser


def _missing_campaign(path: Path, json_mode: bool) -> int:
    payload = {
        "error": f"campaign contract not found: {path}",
        "missing_decisions": [
            "system: model, runtime, hardware, backend, quantization",
            "workload: contexts, batch shape, mode, prompts",
            "objective: metric and maximize/minimize direction",
            "constraints: correctness, quality, memory, power, or other gates",
        ],
        "next": f"create {path} from a benchmark/research contract, then rerun autoluce research",
    }
    if json_mode:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(payload["error"], file=sys.stderr)
        print("AutoLuce needs these human decisions:", file=sys.stderr)
        for decision in payload["missing_decisions"]:
            print(f"  - {decision}", file=sys.stderr)
        print(payload["next"], file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.campaign.exists():
        return _missing_campaign(args.campaign, args.json)
    store = CampaignStore(args.campaign)
    try:
        goal_reference = parse_goal_reference(args.goal).to_dict() if args.goal else None
        against_reference = _load_named_reference(args.against) if args.against else None
        state = store.load()
        if args.record:
            store.record(state, json.loads(args.record.read_text()))
        if goal_reference:
            _append_reference(state, goal_reference)
        if against_reference:
            _append_reference(state, against_reference)
        if args.record or args.goal or args.against:
            # Attachment and collection are valid independently of interpretation.
            # Persist them even when a requested comparison subsequently fails closed.
            store.save(state)
        if args.compare:
            if state["stage"] not in {"explore", "compare"}:
                raise ValueError("compare requires the campaign to be at explore or compare")
            compare_state(state)
            if state["stage"] == "explore":
                validate_lifecycle_transition("explore", "compare", has_reference=bool(state["references"]))
                state["stage"] = state["campaign"]["lifecycle_stage"] = "compare"
                if "compare" not in state["stage_history"]:
                    state["stage_history"].append("compare")
        if args.advance:
            if args.advance == "promote":
                raise ValueError("use --promote EVIDENCE_ID from the explain stage")
            validate_lifecycle_transition(
                state["stage"], args.advance, has_reference=bool(state["references"]),
            )
            state["stage"] = args.advance
            state["campaign"]["lifecycle_stage"] = args.advance
            if not state["stage_history"] or state["stage_history"][-1] != args.advance:
                state["stage_history"].append(args.advance)
        if args.promote:
            if state["stage"] not in {"explain", "promote"}:
                raise ValueError("promote requires the campaign to be at explain or promote")
            if args.promote not in state["frontier"]:
                raise ValueError("only quality-constrained frontier evidence can be promoted")
            promotion_violations = _promotion_constraint_violations(state, args.promote)
            if promotion_violations:
                raise ValueError("promotion constraints failed: " + "; ".join(promotion_violations))
            state["promotion"] = args.promote
            state["stage"] = "promote"
            state["campaign"]["lifecycle_stage"] = "promote"
            if "promote" not in state["stage_history"]:
                state["stage_history"].append("promote")
        store.save(state)
    except (CompatibilityError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if args.json:
            print(json.dumps({"error": str(error)}, sort_keys=True))
        else:
            print(f"autoluce research: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(state, sort_keys=True))
    else:
        reference = "none" if not state["references"] else state["references"][-1]["kind"]
        print(f"Campaign {state['campaign']['name']} ({state['stage']})")
        print(f"Evidence: {len(state['evidence'])}; frontier: {len(state['frontier'])}; reference: {reference}")
        if not state["references"]:
            print("Comparison is optional. Attach one later with --against or --goal.")
    return 0


__all__ = [
    "Campaign", "CampaignEvidence", "CampaignStore", "CompatibilityError",
    "ParetoArchive", "Reference", "compare_state", "migrate_v1_contract",
    "parse_goal_reference",
]


if __name__ == "__main__":
    raise SystemExit(main())
