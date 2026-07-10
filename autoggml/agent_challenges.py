"""Stable public imports for the agent challenge vertical slice."""

from autoggml.agent_domain import AgentJoinRequest, AgentOutput, ChallengeRequest
from autoggml.agent_gate import CandidatePatchGate
from autoggml.agent_repository import FileAgentRepository
from autoggml.agent_runner import AgentRunner, FakeAgentBackend
from autoggml.agent_service import AgentService

__all__ = [
    "AgentJoinRequest",
    "AgentOutput",
    "AgentRunner",
    "AgentService",
    "CandidatePatchGate",
    "ChallengeRequest",
    "FakeAgentBackend",
    "FileAgentRepository",
]
