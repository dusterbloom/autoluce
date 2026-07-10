"""Restricted HTTP transport for the coordination service.

Only typed fleet operations are exposed. The API never accepts shell commands.
"""

from __future__ import annotations

import base64
import hmac
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from autoggml.agent_domain import (
    AgentArtifact,
    AgentContext,
    AgentJoinRequest,
    AgentOutput,
    AgentParticipant,
    AgentSnapshot,
    ArtifactEvidence,
    ChallengeRequest,
    ResearchChallenge,
    artifact_from_dict,
    task_from_dict,
)
from autoggml.agent_service import AgentService
from autoggml.coordination import (
    Candidate,
    CandidateRequest,
    Claim,
    FleetService,
    FleetSnapshot,
    Job,
    JoinRequest,
    Worker,
)


MAX_PATCH_BYTES = 32 * 1024 * 1024


def _snapshot_from_payload(payload: dict[str, Any]) -> FleetSnapshot:
    workers = [Worker(**value) for value in payload["workers"]]
    candidates = [
        Candidate(**{**value, "patch_path": Path(value["patch_path"])})
        for value in payload["candidates"]
    ]
    job_fields = {"job_id", "candidate_id", "worker_id", "status"}
    jobs = [Job(**{key: value[key] for key in job_fields}) for value in payload["jobs"]]
    return FleetSnapshot(workers, candidates, jobs)


class _CoordinatorHttpClient:
    def __init__(self, base_url: str, token: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error", str(error))
            except (json.JSONDecodeError, UnicodeDecodeError):
                message = str(error)
            raise RuntimeError(message) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"coordinator is unavailable: {error.reason}") from error


class CoordinatorClient(_CoordinatorHttpClient):
    """Client-side gateway with the same public operations as FleetService."""

    def join(self, request: JoinRequest) -> Worker:
        return Worker(**self._request("POST", "/v1/join", asdict(request)))

    def submit(self, request: CandidateRequest) -> Job:
        patch = request.patch_path.expanduser().read_bytes()
        if len(patch) > MAX_PATCH_BYTES:
            raise ValueError("candidate patch exceeds the 32 MiB coordinator limit")
        payload = {
            "title": request.title,
            "model": request.model,
            "backends": request.backends,
            "patch_name": request.patch_path.name,
            "patch_base64": base64.b64encode(patch).decode(),
        }
        return Job(**self._request("POST", "/v1/submit", payload))

    def snapshot(self) -> FleetSnapshot:
        return _snapshot_from_payload(self._request("GET", "/v1/status"))

    def pause(self, worker_id: str) -> Worker:
        return Worker(**self._request("POST", f"/v1/workers/{worker_id}/pause"))

    def resume(self, worker_id: str) -> Worker:
        return Worker(**self._request("POST", f"/v1/workers/{worker_id}/resume"))

    def leave(self, worker_id: str) -> None:
        self._request("POST", f"/v1/workers/{worker_id}/leave")

    def claim(self, worker_id: str) -> Claim | None:
        payload = self._request("POST", f"/v1/workers/{worker_id}/claim")
        if payload.get("claim") is None:
            return None
        value = payload["claim"]
        candidate = Candidate(**{**value["candidate"], "patch_path": Path(value["candidate"]["patch_path"])})
        return Claim(Job(**value["job"]), candidate, base64.b64decode(value["patch_base64"]))

    def finish(self, job_id: str, status: str, result: dict[str, Any]) -> Job:
        return Job(**self._request("POST", f"/v1/jobs/{job_id}/finish", {"status": status, "result": result}))


class AgentCoordinatorClient(_CoordinatorHttpClient):
    """Remote gateway implementing the AgentService command surface."""

    def join(self, request: AgentJoinRequest) -> AgentParticipant:
        return AgentParticipant(**self._request("POST", "/v1/agents/join", asdict(request)))

    def create_challenge(self, request: ChallengeRequest) -> ResearchChallenge:
        return ResearchChallenge(**self._request("POST", "/v1/agent-challenges", asdict(request)))

    def next_tasks(self, agent_id: str) -> list:
        payload = self._request("POST", f"/v1/agents/{agent_id}/next")
        return [task_from_dict(value) for value in payload]

    def claim(self, agent_id: str, task_id: str, lease_seconds: int = 1800):
        value = self._request(
            "POST", f"/v1/agents/{agent_id}/tasks/{task_id}/claim", {"lease_seconds": lease_seconds},
        )
        return task_from_dict(value)

    def context(self, agent_id: str, task_id: str) -> AgentContext:
        value = self._request("GET", f"/v1/agents/{agent_id}/tasks/{task_id}/context")
        visible = [
            ArtifactEvidence(artifact_from_dict(item["artifact"]), item["evaluation_result"])
            for item in value["visible_artifacts"]
        ]
        return AgentContext(
            ResearchChallenge(**value["challenge"]), task_from_dict(value["task"]), visible,
        )

    def finish(self, agent_id: str, task_id: str, output: AgentOutput) -> AgentArtifact:
        payload = {
            "rationale": output.rationale,
            "observations": output.observations,
            "risks": output.risks,
            "source_artifact_ids": output.source_artifact_ids,
            "patch_base64": base64.b64encode(output.patch).decode() if output.patch is not None else None,
        }
        return artifact_from_dict(self._request("POST", f"/v1/agents/{agent_id}/tasks/{task_id}/finish", payload))

    def fail(self, agent_id: str, task_id: str, error: Exception) -> AgentArtifact:
        value = self._request(
            "POST", f"/v1/agents/{agent_id}/tasks/{task_id}/fail",
            {"error": str(error), "error_type": type(error).__name__},
        )
        return artifact_from_dict(value)

    def advance(self, challenge_id: str) -> ResearchChallenge:
        return ResearchChallenge(**self._request("POST", f"/v1/agent-challenges/{challenge_id}/advance"))

    def challenge_card(self, challenge_id: str) -> dict:
        return self._request("GET", f"/v1/agent-challenges/{challenge_id}/card")

    def status(self) -> dict:
        return self._request("GET", "/v1/agents/status")

    def snapshot(self) -> AgentSnapshot:
        value = self._request("GET", "/v1/agents/status")
        return AgentSnapshot(
            [AgentParticipant(**item) for item in value["agents"]],
            [ResearchChallenge(**item) for item in value["challenges"]],
            [task_from_dict(item) for item in value["tasks"]],
            [artifact_from_dict(item) for item in value["artifacts"]],
        )


def create_server(
    address: tuple[str, int],
    service: FleetService,
    *,
    token: str,
    upload_dir: Path,
    agent_service: AgentService | None = None,
) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("coordinator token must not be empty")
    upload_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            return hmac.compare_digest(supplied, expected)

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_PATCH_BYTES * 2:
                raise ValueError("request is too large")
            return json.loads(self.rfile.read(length)) if length else {}

        def _dispatch(self) -> dict:
            if self.command == "GET" and self.path == "/v1/status":
                return service.snapshot().to_dict()
            if agent_service is not None and self.command == "GET":
                if self.path == "/v1/agents/status":
                    return agent_service.status()
                if self.path.startswith("/v1/agent-challenges/") and self.path.endswith("/card"):
                    challenge_id = self.path[len("/v1/agent-challenges/"):-len("/card")]
                    return agent_service.challenge_card(challenge_id)
                if self.path.startswith("/v1/agents/") and self.path.endswith("/context"):
                    rest = self.path[len("/v1/agents/"):-len("/context")]
                    agent_id, task_id = rest.split("/tasks/", 1)
                    context = agent_service.context(agent_id, task_id)
                    return {
                        "challenge": asdict(context.challenge),
                        "task": asdict(context.task),
                        "visible_artifacts": [
                            {
                                "artifact": {
                                    **asdict(item.artifact),
                                    "patch_path": str(item.artifact.patch_path) if item.artifact.patch_path else None,
                                },
                                "evaluation_result": item.evaluation_result,
                            }
                            for item in context.visible_artifacts
                        ],
                    }
            if self.command != "POST":
                raise ValueError("unsupported coordinator operation")
            body = self._body()
            if agent_service is not None:
                if self.path == "/v1/agents/join":
                    return asdict(agent_service.join(AgentJoinRequest(**body)))
                if self.path == "/v1/agent-challenges":
                    return asdict(agent_service.create_challenge(ChallengeRequest(**body)))
                if self.path.startswith("/v1/agent-challenges/") and self.path.endswith("/advance"):
                    challenge_id = self.path[len("/v1/agent-challenges/"):-len("/advance")]
                    return asdict(agent_service.advance(challenge_id))
                if self.path.startswith("/v1/agents/"):
                    rest = self.path[len("/v1/agents/"):]
                    if rest.endswith("/next"):
                        return [asdict(task) for task in agent_service.next_tasks(rest[:-len("/next")])]
                    if "/tasks/" in rest:
                        agent_id, task_action = rest.split("/tasks/", 1)
                        task_id, action = task_action.rsplit("/", 1)
                        if action == "claim":
                            return asdict(agent_service.claim(agent_id, task_id, int(body.get("lease_seconds", 1800))))
                        if action == "finish":
                            encoded = body.pop("patch_base64", None)
                            patch = base64.b64decode(encoded, validate=True) if encoded is not None else None
                            if patch is not None and len(patch) > MAX_PATCH_BYTES:
                                raise ValueError("candidate patch exceeds the 32 MiB coordinator limit")
                            artifact = agent_service.finish(agent_id, task_id, AgentOutput(patch=patch, **body))
                            return {**asdict(artifact), "patch_path": str(artifact.patch_path) if artifact.patch_path else None}
                        if action == "fail":
                            error = RuntimeError(body.get("error", "remote agent failed"))
                            artifact = agent_service.fail(agent_id, task_id, error)
                            return {**asdict(artifact), "patch_path": None}
            if self.path == "/v1/join":
                return asdict(service.join(JoinRequest(**body)))
            if self.path == "/v1/submit":
                encoded = body.pop("patch_base64")
                patch = base64.b64decode(encoded, validate=True)
                if len(patch) > MAX_PATCH_BYTES:
                    raise ValueError("candidate patch exceeds the 32 MiB coordinator limit")
                suffix = Path(body.pop("patch_name", "candidate.patch")).suffix or ".patch"
                with tempfile.NamedTemporaryFile(dir=upload_dir, suffix=suffix) as stream:
                    stream.write(patch)
                    stream.flush()
                    job = service.submit(CandidateRequest(patch_path=Path(stream.name), **body))
                return asdict(job)
            job_prefix = "/v1/jobs/"
            if self.path.startswith(job_prefix) and self.path.endswith("/finish"):
                job_id = self.path[len(job_prefix):-len("/finish")]
                return asdict(service.finish(job_id, body["status"], body["result"]))
            prefix = "/v1/workers/"
            if self.path.startswith(prefix):
                worker_id, action = self.path[len(prefix):].rsplit("/", 1)
                if action == "pause":
                    return asdict(service.pause(worker_id))
                if action == "resume":
                    return asdict(service.resume(worker_id))
                if action == "leave":
                    service.leave(worker_id)
                    return {"ok": True}
                if action == "claim":
                    claim = service.claim(worker_id)
                    if claim is None:
                        return {"claim": None}
                    return {"claim": {
                        "job": asdict(claim.job),
                        "candidate": {**asdict(claim.candidate), "patch_path": str(claim.candidate.patch_path)},
                        "patch_base64": base64.b64encode(claim.patch).decode(),
                    }}
            raise ValueError("unsupported coordinator operation")

        def _handle(self) -> None:
            if not self._authorized():
                self._send(401, {"error": "coordinator authentication failed"})
                return
            try:
                self._send(200, self._dispatch())
            except (ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send(409, {"error": str(error)})

        do_GET = _handle
        do_POST = _handle

    return ThreadingHTTPServer(address, Handler)
