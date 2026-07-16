"""Immutable measurement evidence and fail-closed comparison policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from autoluce.research_contract import content_id


SYSTEM_COMPATIBILITY_FIELDS = (
    "machine", "model", "model_fingerprint", "runtime", "hardware", "backend",
    "quantization", "environment",
)


class CompatibilityError(ValueError):
    """Evidence cannot be interpreted as a valid comparison."""

    def __init__(self, mismatches: list[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__("incompatible evidence: " + ", ".join(mismatches))


@dataclass(frozen=True)
class CampaignEvidence:
    campaign_id: str
    system: Mapping[str, Any]
    workload: Mapping[str, Any]
    metrics: Mapping[str, float]
    gates: Mapping[str, bool]
    artifact_hash: str
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def create(cls, **values: Any) -> "CampaignEvidence":
        return cls(**values)

    @property
    def evidence_id(self) -> str:
        return content_id("evidence", self.to_dict(include_id=False))

    @property
    def feasible(self) -> bool:
        return bool(self.gates) and all(bool(value) for value in self.gates.values())

    @property
    def compatibility_key(self) -> str:
        return content_id("compat", {
            "system": {name: self.system.get(name) for name in SYSTEM_COMPATIBILITY_FIELDS},
            "workload": self.workload,
        })

    def compatibility_mismatches(
        self,
        other: "CampaignEvidence",
        *,
        allowed_system_variations: frozenset[str] = frozenset(),
    ) -> list[str]:
        mismatches = []
        for name in SYSTEM_COMPATIBILITY_FIELDS:
            if name in allowed_system_variations:
                continue
            left, right = self.system.get(name), other.system.get(name)
            if left is None or right is None or str(left).startswith("unknown:") or str(right).startswith("unknown:"):
                mismatches.append(name)
            elif left != right:
                mismatches.append(name)
        if self.workload != other.workload:
            mismatches.append("workload")
        return mismatches

    def compare(
        self,
        other: "CampaignEvidence",
        *,
        allowed_system_variations: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        mismatches = self.compatibility_mismatches(
            other, allowed_system_variations=allowed_system_variations,
        )
        if mismatches:
            raise CompatibilityError(mismatches)
        shared = sorted(set(self.metrics) & set(other.metrics))
        return {
            "reference_evidence_id": self.evidence_id,
            "candidate_evidence_id": other.evidence_id,
            "deltas": {metric: float(other.metrics[metric]) - float(self.metrics[metric]) for metric in shared},
            "compatible": True,
        }

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if include_id:
            value["evidence_id"] = self.evidence_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignEvidence":
        raw = dict(value)
        expected = raw.pop("evidence_id", None)
        evidence = cls(**raw)
        if expected is not None and expected != evidence.evidence_id:
            raise ValueError("evidence content does not match evidence_id")
        return evidence
