"""Immutable domain values for cooperative and competitive agent research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TASK_CAPABILITY = {"implement": "implement", "review": "review", "recombine": "recombine"}
TERMINAL_TASK_STATES = {"completed", "failed"}


@dataclass(frozen=True)
class AgentJoinRequest:
    name: str
    capabilities: list[str]
    task_budget: int = 10


@dataclass(frozen=True)
class AgentParticipant:
    agent_id: str
    name: str
    capabilities: list[str]
    task_budget: int
    tasks_claimed: int = 0
    state: str = "ready"


@dataclass(frozen=True)
class ChallengeRequest:
    title: str
    objective: str
    why: str
    evidence: list[str]
    model: str
    backends: list[str]
    base_commit: str = "HEAD"
    implementation_slots: int = 3
    token_budget: int = 20_000
    time_budget_minutes: int = 45
    approaches: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchChallenge:
    challenge_id: str
    title: str
    objective: str
    why: str
    evidence: list[str]
    model: str
    backends: list[str]
    base_commit: str
    implementation_slots: int
    token_budget: int
    time_budget_minutes: int
    approaches: list[str]
    status: str = "building"


@dataclass(frozen=True)
class TaskPacket:
    objective: str
    approach: str
    why: str
    evidence: list[str]
    allowed_paths: list[str]
    forbidden_paths: list[str]
    token_budget: int
    time_budget_minutes: int
    done_when: list[str]
    test_command: str
    expected_impact: str = "high"
    difficulty: str = "medium"
    hardware_validation: list[str] = field(default_factory=list)
    source_artifact_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    challenge_id: str
    kind: str
    packet: TaskPacket
    status: str = "available"
    assigned_agent_id: str | None = None
    lease_expires_at: float | None = None


@dataclass(frozen=True)
class AgentOutput:
    rationale: str
    observations: list[str]
    risks: list[str]
    patch: bytes | None = None
    source_artifact_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    challenge_id: str
    task_id: str
    agent_id: str
    role: str
    rationale: str
    observations: list[str]
    risks: list[str]
    source_artifact_ids: list[str]
    status: str = "completed"
    patch_path: Path | None = None
    candidate_id: str | None = None
    evaluation_job_id: str | None = None


@dataclass(frozen=True)
class ArtifactEvidence:
    artifact: AgentArtifact
    evaluation_result: dict[str, Any] | None


@dataclass(frozen=True)
class AgentContext:
    challenge: ResearchChallenge
    task: AgentTask
    visible_artifacts: list[ArtifactEvidence]


@dataclass(frozen=True)
class AgentSnapshot:
    agents: list[AgentParticipant]
    challenges: list[ResearchChallenge]
    tasks: list[AgentTask]
    artifacts: list[AgentArtifact]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [asdict(value) for value in self.agents],
            "challenges": [asdict(value) for value in self.challenges],
            "tasks": [{**asdict(value), "packet": asdict(value.packet)} for value in self.tasks],
            "artifacts": [
                {**asdict(value), "patch_path": str(value.patch_path) if value.patch_path else None}
                for value in self.artifacts
            ],
        }


def task_from_dict(value: dict[str, Any]) -> AgentTask:
    return AgentTask(**{**value, "packet": TaskPacket(**value["packet"])})


def artifact_from_dict(value: dict[str, Any]) -> AgentArtifact:
    patch_path = Path(value["patch_path"]) if value.get("patch_path") else None
    return AgentArtifact(**{**value, "patch_path": patch_path})
