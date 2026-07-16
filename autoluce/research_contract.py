"""Versioned campaign and performance-reference contracts.

The campaign is the shared human/agent contract.  Performance references are
interpretations attached to that contract; they are deliberately excluded from the
campaign identity so collecting evidence never depends on knowing a competitor.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping


CAMPAIGN_SCHEMA_VERSION = 2
LIFECYCLE = ("observe", "discover", "explore", "compare", "explain", "promote")
LIFECYCLE_TRANSITIONS = {
    "observe": frozenset({"observe", "discover"}),
    "discover": frozenset({"discover", "explore"}),
    "explore": frozenset({"explore", "compare", "explain"}),
    "compare": frozenset({"compare", "explain"}),
    "explain": frozenset({"explain", "promote"}),
    # A campaign is iterative: after promoting one frontier point, a new
    # discover/explore cycle can compare a successor against that accepted result.
    "promote": frozenset({"promote", "discover"}),
}
REFERENCE_KINDS = frozenset({
    "accepted_baseline", "executable", "runtime", "candidate", "result_bundle",
    "measurement", "goal",
})
GOAL_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*(>=|<=|==|>|<)\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class Reference:
    """An optional performance reference, distinct from a quality oracle."""

    kind: str
    locator: str | None = None
    value: float | None = None
    metric: str | None = None
    operator: str | None = None
    provenance: str | None = None
    compatibility: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(f"unsupported reference kind '{self.kind}'")
        if self.kind in {"executable", "runtime", "candidate", "result_bundle"} and not self.locator:
            raise ValueError(f"{self.kind} reference requires locator")
        if self.kind in {"measurement", "goal"} and (not self.metric or self.value is None):
            raise ValueError(f"{self.kind} reference requires metric and value")
        if self.kind == "goal" and self.operator not in {">=", "<=", "==", ">", "<"}:
            raise ValueError("goal reference requires a comparison operator")

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Reference":
        return cls(**dict(value))


def parse_goal_reference(expression: str) -> Reference:
    match = GOAL_PATTERN.fullmatch(expression)
    if not match:
        raise ValueError("goal must look like 'prefill_tok_s >= 1500'")
    metric, operator, raw_value = match.groups()
    return Reference(kind="goal", metric=metric, operator=operator, value=float(raw_value))


def parse_against_reference(value: str) -> Reference:
    normalized = value.strip()
    if normalized in {"baseline", "accepted-baseline", "current", "autoluce"}:
        return Reference(kind="accepted_baseline")
    if normalized.startswith("bundle:"):
        return Reference(kind="result_bundle", locator=normalized.removeprefix("bundle:"))
    if normalized.startswith("candidate:"):
        return Reference(kind="candidate", locator=normalized.removeprefix("candidate:"))
    return Reference(kind="executable", locator=normalized)


def validate_lifecycle_transition(current: str, requested: str, *, has_reference: bool) -> None:
    if current not in LIFECYCLE_TRANSITIONS or requested not in LIFECYCLE:
        raise ValueError(f"unsupported lifecycle transition '{current}' -> '{requested}'")
    if requested not in LIFECYCLE_TRANSITIONS[current]:
        raise ValueError(f"campaign lifecycle cannot jump from {current} to {requested}")
    if requested == "compare" and not has_reference:
        raise ValueError("compare requires a performance reference")


@dataclass(frozen=True)
class Campaign:
    name: str
    system: Mapping[str, Any]
    workload: Mapping[str, Any]
    objective: Mapping[str, Any]
    constraints: Mapping[str, Any]
    reference: Reference | None = None
    evidence: tuple[Any, ...] = field(default_factory=tuple)
    lifecycle_stage: str = "observe"
    schema_version: int = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"unsupported campaign schema {self.schema_version}")
        for name in ("system", "workload", "objective", "constraints"):
            if getattr(self, name) is None:
                raise ValueError(f"campaign requires {name}")
        required_system = {
            "machine", "model", "model_fingerprint", "runtime", "hardware",
            "backend", "quantization", "environment",
        }
        missing_system = sorted(required_system - set(self.system))
        if missing_system:
            raise ValueError("system missing: " + ", ".join(missing_system))
        for name in ("contexts", "batch_shape", "mode", "prompts"):
            if name not in self.workload:
                raise ValueError(f"workload missing: {name}")
        if not self.workload["contexts"] or any(int(value) <= 0 for value in self.workload["contexts"]):
            raise ValueError("workload contexts must be positive")
        if self.objective.get("direction") not in {"maximize", "minimize"}:
            raise ValueError("objective direction must be maximize or minimize")
        if not self.objective.get("metric"):
            raise ValueError("objective requires metric")
        if self.lifecycle_stage not in LIFECYCLE:
            raise ValueError(f"unsupported lifecycle stage '{self.lifecycle_stage}'")

    @property
    def campaign_id(self) -> str:
        return content_id("campaign", {
            "system": self.system,
            "workload": self.workload,
            "objective": self.objective,
            "constraints": self.constraints,
        })

    def attach_reference(self, reference: Reference) -> "Campaign":
        return replace(self, reference=reference)

    def advance(self, stage: str) -> "Campaign":
        validate_lifecycle_transition(
            self.lifecycle_stage, stage, has_reference=self.reference is not None,
        )
        return replace(self, lifecycle_stage=stage)

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "name": self.name,
            "system": dict(self.system),
            "workload": dict(self.workload),
            "objective": dict(self.objective),
            "constraints": dict(self.constraints),
            "reference": self.reference.to_dict() if self.reference else None,
            "lifecycle_stage": self.lifecycle_stage,
        }
        if include_evidence:
            value["evidence"] = [
                item.to_dict() if hasattr(item, "to_dict") else item for item in self.evidence
            ]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Campaign":
        from autoluce.research_evidence import CampaignEvidence

        raw = dict(value)
        raw.pop("campaign_id", None)
        reference = raw.get("reference")
        raw["reference"] = Reference.from_dict(reference) if reference else None
        raw["evidence"] = tuple(CampaignEvidence.from_dict(item) for item in raw.get("evidence", []))
        return cls(**raw)


def migrate_v1_contract(value: Mapping[str, Any]) -> Campaign:
    """Normalize execution contract v1 without treating unknowns as compatible.

    Unknown runtime/hardware/quantization/environment values are explicit.  They make
    migrated campaigns useful for planning while ensuring compatibility checks fail
    against fully identified evidence until the user supplies those identities.
    """

    raw = dict(value)
    version = int(raw.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"unsupported research contract schema {version}")
    objective_map = {
        "interactive_decode": "decode_tok_s",
        "decode_tok_s": "decode_tok_s",
        "prefill_tok_s": "prefill_tok_s",
    }
    primary = str(raw.get("primary_objective", "interactive_decode"))
    if primary not in objective_map:
        raise ValueError(f"cannot migrate v1 primary_objective '{primary}'")
    backend = str(raw.get("primary_backend", "unknown"))
    return Campaign(
        name=f"{raw.get('target', 'legacy')}-{raw.get('model', 'model')}",
        system={
            "machine": str(raw.get("machine_fingerprint", "unknown")),
            "model": str(raw.get("model", "unknown")),
            "model_fingerprint": str(raw.get("model_fingerprint", "unknown")),
            "runtime": "unknown:v1-contract",
            "hardware": "unknown:v1-contract",
            "backend": backend,
            "quantization": "unknown:v1-contract",
            "environment": "unknown:v1-contract",
        },
        workload={
            "name": str(raw.get("workload", "unknown")),
            "contexts": list(raw.get("contexts", [])),
            "batch_shape": {"batch": 1, "ubatch": 1},
            "mode": "decode" if objective_map[primary] == "decode_tok_s" else "prefill",
            "prompts": [],
        },
        objective={"metric": objective_map[primary], "direction": "maximize"},
        constraints={
            "gates": [
                {
                    "metric": "host_headroom_gib",
                    "operator": ">=",
                    "value": float(raw.get("host_headroom_gib", 12.0)),
                },
                {
                    "metric": "accepted_baseline_fraction",
                    "operator": ">=",
                    "value": float(raw.get("baseline_fraction_min", 0.95)),
                    "verification": "comparison_required",
                },
                {
                    "metric": "power_mode",
                    "operator": "==",
                    "value": str(raw.get("power_mode", "unknown")),
                },
            ],
            "quality": {
                "kind": "kl",
                "mean_max": float(raw.get("kl_mean_max", 0.01)),
                "max_max": float(raw.get("kl_max", 0.1)),
            },
        },
    )
