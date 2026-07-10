"""Agent backend interface and one-task runner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from autoluce.agent_domain import AgentContext, AgentOutput, AgentTask
from autoluce.agent_service import AgentService


class AgentBackend(Protocol):
    def execute(self, task: AgentTask, context: AgentContext) -> AgentOutput: ...


class FakeAgentBackend:
    def __init__(self, outputs: list[AgentOutput]) -> None:
        self.outputs = deque(outputs)

    def execute(self, task: AgentTask, context: AgentContext) -> AgentOutput:
        if not self.outputs:
            raise RuntimeError("fake agent has no prepared output")
        return self.outputs.popleft()


@dataclass
class AgentRunner:
    service: AgentService
    agent_id: str
    backend: AgentBackend

    def run_once(self) -> dict:
        choices = self.service.next_tasks(self.agent_id)
        if not choices:
            return {"status": "idle"}
        task = self.service.claim(self.agent_id, choices[0].task_id)
        context = self.service.context(self.agent_id, task.task_id)
        try:
            output = self.backend.execute(task, context)
            artifact = self.service.finish(self.agent_id, task.task_id, output)
            return {"status": "completed", "task_id": task.task_id, "artifact_id": artifact.artifact_id}
        except Exception as error:
            artifact = self.service.fail(self.agent_id, task.task_id, error)
            return {"status": "failed", "task_id": task.task_id, "artifact_id": artifact.artifact_id, "error": str(error)}
