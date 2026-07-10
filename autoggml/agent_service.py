"""Application service orchestrating agent challenges and fleet evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from typing import Callable, Protocol

from autoggml.agent_domain import (
    TASK_CAPABILITY,
    TERMINAL_TASK_STATES,
    AgentArtifact,
    AgentContext,
    AgentJoinRequest,
    AgentOutput,
    AgentParticipant,
    AgentSnapshot,
    AgentTask,
    ArtifactEvidence,
    ChallengeRequest,
    ResearchChallenge,
    TaskPacket,
    task_from_dict,
)
from autoggml.agent_gate import CandidatePatchGate
from autoggml.agent_repository import AgentRepository, serialize_artifact
from autoggml.coordination import CandidateRequest, FleetSnapshot, Job
from autoggml.identifiers import stable_id


class CandidateSubmitter(Protocol):
    def submit(self, request: CandidateRequest) -> Job: ...

    def snapshot(self) -> FleetSnapshot: ...


class AgentService:
    def __init__(
        self,
        repository: AgentRepository,
        candidates: CandidateSubmitter,
        gate: CandidatePatchGate,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.repository = repository
        self.candidates = candidates
        self.gate = gate
        self.clock = clock

    def join(self, request: AgentJoinRequest) -> AgentParticipant:
        capabilities = sorted(set(request.capabilities))
        unknown = set(capabilities) - set(TASK_CAPABILITY.values())
        if not request.name.strip() or not capabilities:
            raise ValueError("agent name and at least one capability are required")
        if unknown:
            raise ValueError("unknown agent capabilities: " + ", ".join(sorted(unknown)))
        if request.task_budget < 1:
            raise ValueError("agent task budget must be positive")
        agent_id = stable_id("agent", request.name.strip())

        def apply(state: dict) -> AgentParticipant:
            for value in state["agents"]:
                if value["agent_id"] == agent_id:
                    return AgentParticipant(**value)
            participant = AgentParticipant(agent_id, request.name.strip(), capabilities, request.task_budget)
            state["agents"].append(asdict(participant))
            return participant

        return self.repository.update(apply)

    def create_challenge(self, request: ChallengeRequest) -> ResearchChallenge:
        if request.implementation_slots < 1:
            raise ValueError("implementation_slots must be positive")
        if not request.backends:
            raise ValueError("challenge requires at least one backend")
        challenge_id = stable_id("challenge", json.dumps(asdict(request), sort_keys=True, separators=(",", ":")))
        challenge = ResearchChallenge(challenge_id=challenge_id, **asdict(request))

        def apply(state: dict) -> ResearchChallenge:
            if any(value["challenge_id"] == challenge_id for value in state["challenges"]):
                return next(ResearchChallenge(**value) for value in state["challenges"] if value["challenge_id"] == challenge_id)
            state["challenges"].append(asdict(challenge))
            for slot in range(request.implementation_slots):
                approach = (
                    request.approaches[slot] if slot < len(request.approaches)
                    else f"independent implementation {slot + 1}"
                )
                packet = self._packet(challenge, approach=approach)
                task = AgentTask(stable_id("task", challenge_id, "implement", str(slot)), challenge_id, "implement", packet)
                state["tasks"].append({**asdict(task), "packet": asdict(task.packet)})
            return challenge

        return self.repository.update(apply)

    def _packet(
        self, challenge: ResearchChallenge, sources: list[str] | None = None, *, approach: str,
    ) -> TaskPacket:
        return TaskPacket(
            objective=challenge.objective,
            approach=approach,
            why=challenge.why,
            evidence=challenge.evidence,
            allowed_paths=["server/src/", "server/include/", "server/deps/llama.cpp/ggml/", "server/CMakeLists.txt"],
            forbidden_paths=[
                "benchmarks/", "autoggml/", "models/", "server/test/", "server/tests/",
                "research contracts", "golden outputs",
            ],
            token_budget=challenge.token_budget,
            time_budget_minutes=challenge.time_budget_minutes,
            done_when=[
                "one focused patch is produced",
                "the stated risk and expected impact are recorded",
                "the backend operator or simulation test passes",
            ],
            test_command="uv run autoggml worker --once --simulate",
            hardware_validation=challenge.backends,
            source_artifact_ids=list(sources or []),
        )

    def _expire_leases(self, state: dict) -> None:
        now = self.clock()
        for index, value in enumerate(state["tasks"]):
            if value["status"] == "claimed" and float(value.get("lease_expires_at") or 0) <= now:
                task = replace(task_from_dict(value), status="available", assigned_agent_id=None, lease_expires_at=None)
                state["tasks"][index] = {**asdict(task), "packet": asdict(task.packet)}

    def next_tasks(self, agent_id: str) -> list[AgentTask]:
        def apply(state: dict) -> list[AgentTask]:
            self._expire_leases(state)
            agent = next((AgentParticipant(**value) for value in state["agents"] if value["agent_id"] == agent_id), None)
            if agent is None:
                raise ValueError(f"unknown agent: {agent_id}")
            if agent.state != "ready" or agent.tasks_claimed >= agent.task_budget:
                return []
            return [
                task_from_dict(value) for value in state["tasks"]
                if value["status"] == "available" and TASK_CAPABILITY[value["kind"]] in agent.capabilities
            ]

        return self.repository.update(apply)

    def claim(self, agent_id: str, task_id: str, lease_seconds: int = 1800) -> AgentTask:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")

        def apply(state: dict) -> AgentTask:
            self._expire_leases(state)
            agent_index = next((i for i, value in enumerate(state["agents"]) if value["agent_id"] == agent_id), None)
            if agent_index is None:
                raise ValueError(f"unknown agent: {agent_id}")
            agent = AgentParticipant(**state["agents"][agent_index])
            if agent.tasks_claimed >= agent.task_budget:
                raise ValueError("agent task budget is exhausted")
            task_index = next((i for i, value in enumerate(state["tasks"]) if value["task_id"] == task_id), None)
            if task_index is None:
                raise ValueError(f"unknown task: {task_id}")
            task = task_from_dict(state["tasks"][task_index])
            if task.status != "available":
                raise ValueError(f"task is not available: {task_id}")
            if TASK_CAPABILITY[task.kind] not in agent.capabilities:
                raise ValueError(f"agent lacks capability for {task.kind}")
            claimed = replace(
                task, status="claimed", assigned_agent_id=agent_id,
                lease_expires_at=self.clock() + lease_seconds,
            )
            state["tasks"][task_index] = {**asdict(claimed), "packet": asdict(claimed.packet)}
            state["agents"][agent_index] = asdict(replace(agent, tasks_claimed=agent.tasks_claimed + 1))
            return claimed

        return self.repository.update(apply)

    def context(self, agent_id: str, task_id: str) -> AgentContext:
        snapshot = self.snapshot()
        task = next((value for value in snapshot.tasks if value.task_id == task_id), None)
        agent = next((value for value in snapshot.agents if value.agent_id == agent_id), None)
        can_preview = (
            task is not None and task.status == "available" and agent is not None
            and TASK_CAPABILITY[task.kind] in agent.capabilities
        )
        if task is None or (task.assigned_agent_id != agent_id and not can_preview):
            raise ValueError("task is neither available nor claimed by this agent")
        challenge = next(value for value in snapshot.challenges if value.challenge_id == task.challenge_id)
        visible = [] if task.kind == "implement" else self._evidence(
            [artifact for artifact in snapshot.artifacts if artifact.artifact_id in task.packet.source_artifact_ids]
        )
        return AgentContext(challenge, task, visible)

    def _evidence(self, artifacts: list[AgentArtifact]) -> list[ArtifactEvidence]:
        jobs = {job.job_id: job for job in self.candidates.snapshot().jobs}
        return [
            ArtifactEvidence(artifact, jobs.get(artifact.evaluation_job_id).result if artifact.evaluation_job_id in jobs else None)
            for artifact in artifacts
        ]

    def finish(self, agent_id: str, task_id: str, output: AgentOutput) -> AgentArtifact:
        context = self.context(agent_id, task_id)
        task = context.task
        visible_ids = {item.artifact.artifact_id for item in context.visible_artifacts}
        if not set(output.source_artifact_ids).issubset(visible_ids):
            raise ValueError("output references artifacts not visible to this task")
        if task.kind in {"implement", "recombine"} and output.patch is None:
            raise ValueError(f"{task.kind} task requires a patch")
        if task.kind == "review" and output.patch is not None:
            raise ValueError("review task must not submit a patch")
        if task.kind == "recombine" and len(output.source_artifact_ids) < 2:
            raise ValueError("recombine task must credit at least two source artifacts")

        patch_sha = hashlib.sha256(output.patch or b"").hexdigest()
        artifact_id = stable_id("artifact", task_id, agent_id, patch_sha, output.rationale)
        patch_path = None
        job = None
        if output.patch is not None:
            self.gate.validate(output.patch)
            patch_path = self.repository.write_patch(artifact_id, output.patch)
            challenge = context.challenge
            job = self.candidates.submit(CandidateRequest(
                f"{challenge.title}: {task.kind} by {agent_id}", patch_path, challenge.backends, challenge.model,
            ))
        artifact = AgentArtifact(
            artifact_id, task.challenge_id, task_id, agent_id, task.kind, output.rationale,
            list(output.observations), list(output.risks), list(output.source_artifact_ids),
            patch_path=patch_path, candidate_id=job.candidate_id if job else None,
            evaluation_job_id=job.job_id if job else None,
        )

        def apply(state: dict) -> AgentArtifact:
            task_index = next(i for i, value in enumerate(state["tasks"]) if value["task_id"] == task_id)
            current = task_from_dict(state["tasks"][task_index])
            if current.status != "claimed" or current.assigned_agent_id != agent_id:
                raise ValueError("task claim changed before submission")
            completed = replace(current, status="completed", lease_expires_at=None)
            state["tasks"][task_index] = {**asdict(completed), "packet": asdict(completed.packet)}
            state["artifacts"].append(serialize_artifact(artifact))
            return artifact

        return self.repository.update(apply)

    def fail(self, agent_id: str, task_id: str, error: Exception) -> AgentArtifact:
        context = self.context(agent_id, task_id)
        task = context.task
        artifact = AgentArtifact(
            stable_id("artifact", task_id, agent_id, "failed"), task.challenge_id, task_id, agent_id, task.kind,
            "Agent task failed", [f"Agent execution failed: {error}"], [type(error).__name__], [], status="failed",
        )

        def apply(state: dict) -> AgentArtifact:
            index = next(i for i, value in enumerate(state["tasks"]) if value["task_id"] == task_id)
            failed = replace(task_from_dict(state["tasks"][index]), status="failed", lease_expires_at=None)
            state["tasks"][index] = {**asdict(failed), "packet": asdict(failed.packet)}
            state["artifacts"].append(serialize_artifact(artifact))
            return artifact

        return self.repository.update(apply)

    def advance(self, challenge_id: str) -> ResearchChallenge:
        snapshot = self.snapshot()
        challenge = next(value for value in snapshot.challenges if value.challenge_id == challenge_id)
        tasks = [task for task in snapshot.tasks if task.challenge_id == challenge_id]
        artifacts = [artifact for artifact in snapshot.artifacts if artifact.challenge_id == challenge_id]
        jobs = {job.job_id: job for job in self.candidates.snapshot().jobs}

        if challenge.status == "building":
            implementations = [task for task in tasks if task.kind == "implement"]
            implementation_artifacts = [artifact for artifact in artifacts if artifact.role == "implement" and artifact.status == "completed"]
            implementations_done = all(task.status in TERMINAL_TASK_STATES for task in implementations)
            if implementations_done and not implementation_artifacts:
                return self._set_challenge_status(challenge_id, "inconclusive")
            evaluations_done = implementation_artifacts and all(
                jobs.get(artifact.evaluation_job_id) and jobs[artifact.evaluation_job_id].status in {"completed", "failed"}
                for artifact in implementation_artifacts
            )
            if implementations_done and evaluations_done:
                return self._add_stage_task(challenge, "review", implementation_artifacts, "reviewing")
        elif challenge.status == "reviewing":
            reviews = [task for task in tasks if task.kind == "review"]
            if reviews and all(task.status in TERMINAL_TASK_STATES for task in reviews):
                sources = [artifact for artifact in artifacts if artifact.status == "completed"]
                return self._add_stage_task(challenge, "recombine", sources, "recombining")
        elif challenge.status == "recombining":
            recombinations = [task for task in tasks if task.kind == "recombine"]
            recombined = [artifact for artifact in artifacts if artifact.role == "recombine" and artifact.status == "completed"]
            recombinations_done = all(task.status in TERMINAL_TASK_STATES for task in recombinations)
            if recombinations and recombinations_done and not recombined:
                return self._set_challenge_status(challenge_id, "inconclusive")
            evaluations_done = recombined and all(
                jobs.get(artifact.evaluation_job_id) and jobs[artifact.evaluation_job_id].status in {"completed", "failed"}
                for artifact in recombined
            )
            if recombinations and recombinations_done and evaluations_done:
                return self._set_challenge_status(challenge_id, "complete")
        return challenge

    def _add_stage_task(
        self, challenge: ResearchChallenge, kind: str, sources: list[AgentArtifact], status: str,
    ) -> ResearchChallenge:
        source_ids = [artifact.artifact_id for artifact in sources]
        approach = "compare measured candidates" if kind == "review" else "combine compatible verified strengths"
        task = AgentTask(
            stable_id("task", challenge.challenge_id, kind), challenge.challenge_id, kind,
            self._packet(challenge, source_ids, approach=approach),
        )

        def apply(state: dict) -> ResearchChallenge:
            if not any(value["task_id"] == task.task_id for value in state["tasks"]):
                state["tasks"].append({**asdict(task), "packet": asdict(task.packet)})
            index = next(i for i, value in enumerate(state["challenges"]) if value["challenge_id"] == challenge.challenge_id)
            updated = replace(ResearchChallenge(**state["challenges"][index]), status=status)
            state["challenges"][index] = asdict(updated)
            return updated

        return self.repository.update(apply)

    def _set_challenge_status(self, challenge_id: str, status: str) -> ResearchChallenge:
        def apply(state: dict) -> ResearchChallenge:
            index = next(i for i, value in enumerate(state["challenges"]) if value["challenge_id"] == challenge_id)
            challenge = replace(ResearchChallenge(**state["challenges"][index]), status=status)
            state["challenges"][index] = asdict(challenge)
            return challenge

        return self.repository.update(apply)

    def challenge_card(self, challenge_id: str) -> dict:
        snapshot = self.snapshot()
        challenge = next(value for value in snapshot.challenges if value.challenge_id == challenge_id)
        agents = {agent.agent_id: agent for agent in snapshot.agents}
        artifacts = [artifact for artifact in snapshot.artifacts if artifact.challenge_id == challenge_id]
        evidence = [] if challenge.status == "building" else self._evidence(artifacts)
        leaderboard = []
        for item in evidence:
            if item.artifact.candidate_id:
                leaderboard.append({
                    "artifact_id": item.artifact.artifact_id,
                    "agent": agents[item.artifact.agent_id].name,
                    "role": item.artifact.role,
                    "score": (item.evaluation_result or {}).get("score"),
                    "correctness": (item.evaluation_result or {}).get("correctness"),
                    "rationale": item.artifact.rationale,
                })
        scored = [entry for entry in leaderboard if isinstance(entry["score"], (int, float))]
        winner = max(scored, key=lambda entry: entry["score"]) if scored else None
        credits = [
            {"agent": agents[artifact.agent_id].name, "role": artifact.role, "artifact_id": artifact.artifact_id,
             "sources": artifact.source_artifact_ids}
            for artifact in artifacts
        ]
        return {
            "challenge_id": challenge_id,
            "title": challenge.title,
            "status": challenge.status,
            "winner": winner,
            "leaderboard": sorted(leaderboard, key=lambda entry: entry["score"] or float("-inf"), reverse=True),
            "credits": [] if challenge.status == "building" else credits,
        }

    def status(self) -> dict:
        snapshot = self.snapshot()
        hidden = {
            challenge.challenge_id for challenge in snapshot.challenges if challenge.status == "building"
        }
        payload = snapshot.to_dict()
        payload["artifacts"] = [
            artifact for artifact in payload["artifacts"] if artifact["challenge_id"] not in hidden
        ]
        payload["summary"] = {
            "agents": len(snapshot.agents),
            "challenges": len(snapshot.challenges),
            "available_tasks": sum(task.status == "available" for task in snapshot.tasks),
            "claimed_tasks": sum(task.status == "claimed" for task in snapshot.tasks),
            "released_artifacts": len(payload["artifacts"]),
        }
        return payload

    def snapshot(self) -> AgentSnapshot:
        return self.repository.snapshot()
