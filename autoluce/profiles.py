"""Canonical machine/model profiles and stable fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


VOLATILE_MACHINE_FIELDS = {
    "timestamp_utc", "mem_available_bytes", "mem_free_bytes", "swap_free_bytes", "load_average",
    "temperature_c", "power_w", "gpu_busy_percent", "active_processes", "busy_reasons", "models",
}


def stable_fingerprint(data: dict[str, Any], volatile: set[str] | None = None) -> str:
    ignored = VOLATILE_MACHINE_FIELDS if volatile is None else volatile
    stable = {key: value for key, value in data.items() if key not in ignored}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class MachineProfile:
    observed: dict[str, Any]
    inferred: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    schema_version: int = 1

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.observed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "observed": self.observed,
            "inferred": self.inferred,
            "unknown": self.unknown,
        }


@dataclass
class ModelProfile:
    model_id: str
    quant: str
    files: list[str]
    size_bytes: int
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.to_dict(), volatile=set())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "quant": self.quant,
            "files": self.files,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }
