"""Persistence boundary for agent participants, tasks, and research artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol, TypeVar

from autoggml.agent_domain import (
    AgentArtifact,
    AgentParticipant,
    AgentSnapshot,
    ResearchChallenge,
    artifact_from_dict,
    task_from_dict,
)
from autoggml.atomic_store import AtomicJsonStore


T = TypeVar("T")


def _default_state() -> dict:
    return {"schema_version": 1, "agents": [], "challenges": [], "tasks": [], "artifacts": []}


class AgentRepository(Protocol):
    def update(self, operation: Callable[[dict], T]) -> T: ...

    def snapshot(self) -> AgentSnapshot: ...

    def write_patch(self, artifact_id: str, patch: bytes) -> Path: ...


class FileAgentRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = AtomicJsonStore(root / "state.json", _default_state)
        self.artifacts_dir = root / "artifacts"

    def update(self, operation: Callable[[dict], T]) -> T:
        return self.store.update(operation)

    def snapshot(self) -> AgentSnapshot:
        return self.update(lambda state: AgentSnapshot(
            agents=[AgentParticipant(**value) for value in state["agents"]],
            challenges=[ResearchChallenge(**value) for value in state["challenges"]],
            tasks=[task_from_dict(value) for value in state["tasks"]],
            artifacts=[artifact_from_dict(value) for value in state["artifacts"]],
        ))

    def write_patch(self, artifact_id: str, patch: bytes) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"{artifact_id}.patch"
        path.write_bytes(patch)
        return path


def serialize_artifact(artifact: AgentArtifact) -> dict:
    return {**asdict(artifact), "patch_path": str(artifact.patch_path) if artifact.patch_path else None}
