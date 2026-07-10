"""Versioned research-contract schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResearchContract:
    target: str
    machine_fingerprint: str
    model: str
    model_fingerprint: str
    primary_objective: str = "interactive_decode"
    workload: str = "interactive_single_user"
    contexts: list[int] = field(default_factory=lambda: [8192, 32768, 131072])
    backends: list[str] = field(default_factory=lambda: ["hip", "vulkan"])
    primary_backend: str = "hip"
    host_headroom_gib: float = 12.0
    power_mode: str = "maximum_performance"
    kl_mean_max: float = 0.01
    kl_max: float = 0.1
    baseline_fraction_min: float = 0.95
    build_jobs: int = 4
    confirmed_assumptions: list[str] = field(default_factory=list)
    schema_version: int = 1

    def validate(self) -> None:
        if self.primary_backend not in self.backends:
            raise ValueError("primary_backend must be allowed by backends")
        if not self.contexts or any(ctx <= 0 for ctx in self.contexts):
            raise ValueError("contexts must contain positive token depths")
        if not 1 <= self.build_jobs <= 4:
            raise ValueError("build_jobs must be between 1 and 4")
        if not 0 < self.baseline_fraction_min <= 1:
            raise ValueError("baseline_fraction_min must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def read(cls, path: Path) -> "ResearchContract":
        value = cls(**yaml.safe_load(path.read_text()))
        value.validate()
        return value
