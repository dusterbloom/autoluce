"""Agent challenge behavior, written before the implementation."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace

import pytest

from autoluce.agent_challenges import (
    AgentJoinRequest,
    AgentOutput,
    AgentRunner,
    AgentService,
    CandidatePatchGate,
    ChallengeRequest,
    FakeAgentBackend,
    FileAgentRepository,
)
from autoluce.agent_cli import _source_commit, _source_root
from autoluce.coordination import FileCoordinationRepository, FleetService, JoinRequest
from autoluce.coordinator_http import AgentCoordinatorClient, create_server
from autoluce.team_worker import TeamWorker


def _patch(symbol: str) -> bytes:
    return (
        f"diff --git a/server/src/{symbol}.cpp b/server/src/{symbol}.cpp\n"
        "--- a/server/src/old.cpp\n"
        "+++ b/server/src/new.cpp\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    ).encode()


def _services(tmp_path: Path, clock=lambda: 1000.0):
    fleet = FleetService(FileCoordinationRepository(tmp_path / "fleet"))
    agents = AgentService(FileAgentRepository(tmp_path / "agents"), fleet, CandidatePatchGate(), clock=clock)
    return agents, fleet


def _challenge(slots: int = 2) -> ChallengeRequest:
    return ChallengeRequest(
        title="Sinkhorn challenge",
        objective="Reduce batch-one Sinkhorn dispatch overhead",
        why="Profiler attributes 29.4% of decode kernel time to tiny operations.",
        evidence=["rocprof capture rp-17", "baseline 10.3 tok/s"],
        model="deepseek-v4-flash",
        backends=["hip"],
        implementation_slots=slots,
        token_budget=20_000,
        time_budget_minutes=45,
        approaches=["kernel fusion", "persistent buffer reuse"][:slots],
    )


def test_agents_choose_capability_matched_tasks_and_implementations_are_blind(tmp_path):
    service, _ = _services(tmp_path)
    ada = service.join(AgentJoinRequest("ada", ["implement"], task_budget=2))
    reviewer = service.join(AgentJoinRequest("turing", ["review"], task_budget=2))
    challenge = service.create_challenge(_challenge())

    choices = service.next_tasks(ada.agent_id)
    assert len(choices) == 2
    assert all(task.kind == "implement" for task in choices)
    assert service.next_tasks(reviewer.agent_id) == []
    task = service.claim(ada.agent_id, choices[0].task_id)
    context = service.context(ada.agent_id, task.task_id)

    assert context.challenge.challenge_id == challenge.challenge_id
    assert context.task.packet.objective == _challenge().objective
    assert {choice.packet.approach for choice in choices} == {"kernel fusion", "persistent buffer reuse"}
    assert context.task.packet.done_when
    assert context.visible_artifacts == []


def test_expired_task_lease_returns_to_available_pool(tmp_path):
    now = [1000.0]
    service, _ = _services(tmp_path, clock=lambda: now[0])
    first = service.join(AgentJoinRequest("first", ["implement"], task_budget=2))
    second = service.join(AgentJoinRequest("second", ["implement"], task_budget=2))
    service.create_challenge(_challenge(slots=1))
    task = service.next_tasks(first.agent_id)[0]
    service.claim(first.agent_id, task.task_id, lease_seconds=30)

    assert service.next_tasks(second.agent_id) == []
    now[0] += 31
    choices = service.next_tasks(second.agent_id)

    assert [choice.task_id for choice in choices] == [task.task_id]
    assert service.claim(second.agent_id, task.task_id).assigned_agent_id == second.agent_id


def test_agent_task_budget_is_enforced(tmp_path):
    service, _ = _services(tmp_path)
    agent = service.join(AgentJoinRequest("bounded", ["implement"], task_budget=1))
    service.create_challenge(_challenge(slots=2))
    first, second = service.next_tasks(agent.agent_id)
    service.claim(agent.agent_id, first.task_id)

    with pytest.raises(ValueError, match="task budget"):
        service.claim(agent.agent_id, second.task_id)


def test_challenge_identity_changes_with_the_research_contract(tmp_path):
    service, _ = _services(tmp_path)
    original = _challenge(slots=1)
    changed = replace(original, approaches=["different kernel strategy"])

    assert service.create_challenge(original).challenge_id != service.create_challenge(changed).challenge_id


def test_candidate_gate_allows_engine_patch_and_rejects_research_or_traversal_changes():
    gate = CandidatePatchGate()
    assert gate.validate(_patch("sinkhorn")) == ["server/src/sinkhorn.cpp"]
    vendor = b"diff --git a/server/deps/llama.cpp/ggml/src/op.cpp b/server/deps/llama.cpp/ggml/src/op.cpp\n"
    assert gate.validate(vendor) == ["server/deps/llama.cpp/ggml/src/op.cpp"]

    protected = b"diff --git a/benchmarks/golden.json b/benchmarks/golden.json\n"
    legacy_root = b"diff --git a/src/old.cpp b/src/old.cpp\n"
    traversal = b"diff --git a/server/src/../../quality.py b/server/src/../../quality.py\n"
    with pytest.raises(ValueError, match="not an approved Lucebox product path"):
        gate.validate(protected)
    with pytest.raises(ValueError, match="unsafe patch path"):
        gate.validate(traversal)
    with pytest.raises(ValueError, match="not an approved Lucebox product path"):
        gate.validate(legacy_root)


def test_agent_workspace_uses_the_pinned_product_checkout(monkeypatch, tmp_path):
    source = tmp_path / "work" / "lucebox"
    source.mkdir(parents=True)
    (tmp_path / "work" / "lucebox.pin").write_text("abc123\n")
    monkeypatch.setattr("autoluce.agent_cli.ROOT", tmp_path)
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(source))
    monkeypatch.delenv("AUTOLUCE_SOURCE_COMMIT", raising=False)

    assert _source_root() == source
    assert _source_commit() == "abc123"


def test_parallel_agents_compete_then_review_and_recombine_with_shared_credit(tmp_path):
    service, fleet = _services(tmp_path)
    machine_a = fleet.join(JoinRequest("lucebox3", "box-3", ["hip"], 128.0))
    machine_b = fleet.join(JoinRequest("lucebox5", "box-5", ["hip"], 128.0))
    ada = service.join(AgentJoinRequest("ada", ["implement"], task_budget=2))
    hopper = service.join(AgentJoinRequest("hopper", ["implement"], task_budget=2))
    turing = service.join(AgentJoinRequest("turing", ["review"], task_budget=2))
    grace = service.join(AgentJoinRequest("grace", ["recombine"], task_budget=2))
    challenge = service.create_challenge(_challenge())

    ada_result = AgentRunner(
        service, ada.agent_id,
        FakeAgentBackend([AgentOutput("Fuse the graph op", ["fewer launches"], ["register pressure"], _patch("fused"))]),
    ).run_once()
    hopper_result = AgentRunner(
        service, hopper.agent_id,
        FakeAgentBackend([AgentOutput("Reuse a persistent buffer", ["less allocation"], ["more state"], _patch("buffer"))]),
    ).run_once()
    assert ada_result["status"] == hopper_result["status"] == "completed"

    # Initial implementation contexts remain blind even after a competing result lands.
    implementation_artifacts = service.snapshot().artifacts
    assert len(implementation_artifacts) == 2
    assert all(not artifact.source_artifact_ids for artifact in implementation_artifacts)
    assert service.status()["artifacts"] == []
    assert service.challenge_card(challenge.challenge_id)["leaderboard"] == []

    TeamWorker(fleet, machine_a.worker_id, lambda _: {"score": 1.12, "correctness": "pass"}).run_once()
    TeamWorker(fleet, machine_b.worker_id, lambda _: {"score": 1.08, "correctness": "pass"}).run_once()
    service.advance(challenge.challenge_id)

    review_choices = service.next_tasks(turing.agent_id)
    assert len(review_choices) == 1
    review_context = service.context(turing.agent_id, review_choices[0].task_id)
    assert {item.artifact.agent_id for item in review_context.visible_artifacts} == {ada.agent_id, hopper.agent_id}
    assert {item.evaluation_result["score"] for item in review_context.visible_artifacts} == {1.12, 1.08}
    review = AgentRunner(
        service, turing.agent_id,
        FakeAgentBackend([AgentOutput(
            "Combine fusion with the persistent layout", ["approaches are compatible"], [], None,
            [item.artifact.artifact_id for item in review_context.visible_artifacts],
        )]),
    ).run_once()
    assert review["status"] == "completed"
    service.advance(challenge.challenge_id)

    recombine_choices = service.next_tasks(grace.agent_id)
    assert len(recombine_choices) == 1
    combined = AgentRunner(
        service, grace.agent_id,
        FakeAgentBackend([AgentOutput(
            "Hybrid fusion and buffer layout", ["retains both benefits"], ["larger patch"], _patch("hybrid"),
            [artifact.artifact_id for artifact in implementation_artifacts],
        )]),
    ).run_once()
    assert combined["status"] == "completed"
    TeamWorker(fleet, machine_a.worker_id, lambda _: {"score": 1.21, "correctness": "pass"}).run_once()
    completed = service.advance(challenge.challenge_id)

    assert completed.status == "complete"
    card = service.challenge_card(challenge.challenge_id)
    assert card["winner"]["score"] == 1.21
    assert card["winner"]["agent"] == "grace"
    assert {credit["agent"] for credit in card["credits"]} == {"ada", "hopper", "turing", "grace"}
    assert any(entry["score"] == 1.08 for entry in card["leaderboard"])


def test_agent_runner_records_failure_as_research_evidence(tmp_path):
    service, _ = _services(tmp_path)
    agent = service.join(AgentJoinRequest("ada", ["implement"], task_budget=1))
    service.create_challenge(_challenge(slots=1))

    class BrokenBackend:
        def execute(self, task, context):
            raise RuntimeError("context exhausted")

    result = AgentRunner(service, agent.agent_id, BrokenBackend()).run_once()

    assert result["status"] == "failed"
    artifact = service.snapshot().artifacts[0]
    assert artifact.status == "failed"
    assert artifact.observations == ["Agent execution failed: context exhausted"]
    challenge = service.snapshot().challenges[0]
    assert service.advance(challenge.challenge_id).status == "inconclusive"
    assert service.challenge_card(challenge.challenge_id)["credits"][0]["agent"] == "ada"


def test_agent_http_client_can_choose_claim_and_submit_a_candidate(tmp_path):
    service, fleet = _services(tmp_path)
    fleet.join(JoinRequest("lucebox5", "box-5", ["hip"], 128.0))
    server = create_server(
        ("127.0.0.1", 0), fleet, token="secret", upload_dir=tmp_path / "uploads", agent_service=service,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = AgentCoordinatorClient(f"http://127.0.0.1:{server.server_port}", "secret")
        agent = client.join(AgentJoinRequest("remote-codex", ["implement"], 2))
        challenge = client.create_challenge(_challenge(slots=1))
        task = client.next_tasks(agent.agent_id)[0]
        assert client.context(agent.agent_id, task.task_id).visible_artifacts == []
        client.claim(agent.agent_id, task.task_id)
        artifact = client.finish(
            agent.agent_id, task.task_id,
            AgentOutput("remote patch", ["typed API"], [], _patch("remote")),
        )

        assert artifact.challenge_id == challenge.challenge_id
        assert artifact.candidate_id
        assert client.status()["artifacts"] == []
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _cli(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AUTOLUCE_COORDINATION_DIR"] = str(state)
    env["AUTOLUCE_AGENT_CONFIG"] = str(state / "agent.json")
    env["AUTOLUCE_SOURCE_COMMIT"] = "test-product-pin"
    return subprocess.run(
        [sys.executable, "-m", "cli", *args], text=True, capture_output=True, env=env, check=False,
    )


def test_agent_cli_end_to_end_from_choice_to_simulated_hardware_result(tmp_path):
    state = tmp_path / "team"
    joined_machine = _cli(
        state, "join", "--name", "lucebox5", "--machine-id", "box-5", "--backend", "hip", "--memory-gib", "128",
    )
    assert joined_machine.returncode == 0, joined_machine.stderr
    joined_agent = _cli(
        state, "agent", "join", "--name", "codex-one", "--capability", "implement", "--json",
    )
    assert joined_agent.returncode == 0, joined_agent.stderr
    assert json.loads(joined_agent.stdout)["agent_id"].startswith("agent-")
    assert (state / "agent.json").stat().st_mode & 0o077 == 0
    created = _cli(
        state, "agent", "challenge", "create", "--title", "CLI challenge",
        "--objective", "Remove dispatch overhead", "--why", "profile evidence", "--evidence", "trace-1",
        "--model", "deepseek-v4-flash", "--backend", "hip", "--slots", "1", "--json",
    )
    assert created.returncode == 0, created.stderr
    challenge_id = json.loads(created.stdout)["challenge_id"]
    choices = _cli(state, "agent", "next", "--json")
    task_id = json.loads(choices.stdout)[0]["task_id"]
    started = _cli(state, "agent", "start", task_id, "--no-worktree", "--json")
    assert started.returncode == 0, started.stderr
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(_patch("cli"))
    submitted = _cli(
        state, "agent", "submit", task_id, "--patch", str(patch),
        "--rationale", "one focused candidate", "--observation", "launches reduced", "--risk", "registers", "--json",
    )
    assert submitted.returncode == 0, submitted.stderr
    worker_id = json.loads(_cli(state, "status", "--json").stdout)["workers"][0]["worker_id"]
    assert _cli(state, "worker", "--worker", worker_id, "--once", "--simulate").returncode == 0
    advanced = _cli(state, "agent", "advance", challenge_id, "--json")

    assert advanced.returncode == 0, advanced.stderr
    assert json.loads(advanced.stdout)["status"] == "reviewing"
    status = json.loads(_cli(state, "agent", "status", "--json").stdout)
    assert status["challenges"][0]["status"] == "reviewing"
    assert status["artifacts"][0]["candidate_id"]
