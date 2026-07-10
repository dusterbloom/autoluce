"""Coordination behavior: test the public workflow before its implementation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from autoggml.coordination import (
    CandidateRequest,
    FileCoordinationRepository,
    FleetService,
    JoinRequest,
    NoCompatibleWorkerError,
)
from autoggml.coordinator_http import CoordinatorClient, create_server
from autoggml.team_worker import TeamWorker


def _service(tmp_path: Path) -> FleetService:
    return FleetService(FileCoordinationRepository(tmp_path / "fleet"))


def test_join_is_idempotent_per_physical_machine_and_merges_backends(tmp_path):
    service = _service(tmp_path)

    first = service.join(JoinRequest("peppi-3090", "host-123", ["cuda"], 24.0))
    second = service.join(JoinRequest("peppi-both", "host-123", ["hip", "cuda"], 64.0))

    assert second.worker_id == first.worker_id
    assert second.name == "peppi-both"
    assert second.backends == ["cuda", "hip"]
    assert len(service.snapshot().workers) == 1


def test_submit_routes_to_idle_compatible_worker_and_preserves_candidate(tmp_path):
    patch = tmp_path / "sinkhorn.patch"
    patch.write_text("diff --git a/a b/a\n")
    service = _service(tmp_path)
    cuda = service.join(JoinRequest("local-3090", "machine-cuda", ["cuda"], 24.0))
    hip = service.join(JoinRequest("lucebox5", "machine-hip", ["hip", "vulkan"], 128.0))

    job = service.submit(CandidateRequest("Fuse Sinkhorn", patch, ["hip"], "deepseek-v4-flash"))
    snapshot = service.snapshot()

    assert job.worker_id == hip.worker_id
    assert job.worker_id != cuda.worker_id
    assert snapshot.jobs[0].status == "queued"
    assert snapshot.candidates[0].candidate_id == job.candidate_id
    assert snapshot.candidates[0].patch_sha256
    assert snapshot.candidates[0].patch_path.read_text() == patch.read_text()


def test_candidate_identity_is_content_addressed_and_duplicate_submit_reuses_it(tmp_path):
    patch = tmp_path / "candidate.patch"
    patch.write_text("same experiment\n")
    service = _service(tmp_path)
    service.join(JoinRequest("lucebox3", "machine-3", ["hip"], 128.0))

    a = service.submit(CandidateRequest("First wording", patch, ["hip"], "deepseek-v4-flash"))
    b = service.submit(CandidateRequest("Second wording", patch, ["hip"], "deepseek-v4-flash"))

    assert a.candidate_id == b.candidate_id
    assert len(service.snapshot().candidates) == 1
    assert len(service.snapshot().jobs) == 1


def test_pause_excludes_worker_until_resume(tmp_path):
    patch = tmp_path / "candidate.patch"
    patch.write_text("experiment\n")
    service = _service(tmp_path)
    worker = service.join(JoinRequest("lucebox5", "machine-5", ["hip"], 128.0))
    service.pause(worker.worker_id)

    with pytest.raises(NoCompatibleWorkerError, match="No idle machine supports hip"):
        service.submit(CandidateRequest("Experiment", patch, ["hip"], "deepseek-v4-flash"))

    service.resume(worker.worker_id)
    assert service.submit(CandidateRequest("Experiment", patch, ["hip"], "deepseek-v4-flash")).worker_id == worker.worker_id


def test_leave_refuses_worker_with_assigned_work(tmp_path):
    patch = tmp_path / "candidate.patch"
    patch.write_text("experiment\n")
    service = _service(tmp_path)
    worker = service.join(JoinRequest("lucebox5", "machine-5", ["hip"], 128.0))
    service.submit(CandidateRequest("Experiment", patch, ["hip"], "deepseek-v4-flash"))

    with pytest.raises(ValueError, match="assigned work"):
        service.leave(worker.worker_id)


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AUTOGGML_COORDINATION_DIR"] = str(repo)
    return subprocess.run(
        [sys.executable, "-m", "cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_end_to_end_cli_join_submit_status_pause_resume(tmp_path):
    repo = tmp_path / "team"
    patch = tmp_path / "candidate.patch"
    patch.write_text("diff --git a/a b/a\n")

    joined = _cli(repo, "join", "--name", "lucebox5", "--machine-id", "box-5", "--backend", "hip", "--memory-gib", "128")
    assert joined.returncode == 0, joined.stderr
    assert "lucebox5 is ready" in joined.stdout

    submitted = _cli(repo, "submit", str(patch), "--title", "Fuse Sinkhorn", "--backend", "hip", "--model", "deepseek-v4-flash")
    assert submitted.returncode == 0, submitted.stderr
    assert "Queued on lucebox5" in submitted.stdout

    status = _cli(repo, "status", "--json")
    assert status.returncode == 0, status.stderr
    payload = json.loads(status.stdout)
    assert payload["summary"] == {"machines": 1, "ready": 0, "running": 0, "queued": 1}
    assert payload["jobs"][0]["title"] == "Fuse Sinkhorn"

    worker_id = payload["workers"][0]["worker_id"]
    paused = _cli(repo, "pause", "--worker", worker_id)
    assert paused.returncode == 0, paused.stderr
    resumed = _cli(repo, "resume", "--worker", worker_id)
    assert resumed.returncode == 0, resumed.stderr
    patch_dir = Path(__file__).parent.parent / "patches"
    patches_before = set(patch_dir.glob("team-*.patch"))
    worked = _cli(repo, "worker", "--worker", worker_id, "--once", "--simulate")
    assert worked.returncode == 0, worked.stderr
    final_status = json.loads(_cli(repo, "status", "--json").stdout)
    assert final_status["jobs"][0]["status"] == "completed"
    assert set(patch_dir.glob("team-*.patch")) == patches_before


def test_http_coordinator_supports_the_same_end_to_end_flow(tmp_path):
    service = _service(tmp_path)
    server = create_server(("127.0.0.1", 0), service, token="team-secret", upload_dir=tmp_path / "uploads")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        client = CoordinatorClient(url, "team-secret")
        worker = client.join(JoinRequest("lucebox3", "machine-3", ["hip"], 128.0))
        patch = tmp_path / "candidate.patch"
        patch.write_text("experiment over HTTP\n")
        job = client.submit(CandidateRequest("HTTP candidate", patch, ["hip"], "deepseek-v4-flash"))

        assert job.worker_id == worker.worker_id
        assert client.snapshot().candidates[0].patch_sha256
        claim = client.claim(worker.worker_id)
        assert claim is not None
        assert claim.patch == patch.read_bytes()
        assert client.finish(job.job_id, "completed", {"verdict": "kept"}).result == {"verdict": "kept"}
        assert client.pause(worker.worker_id).state == "paused"
        assert client.resume(worker.worker_id).state == "ready"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_http_coordinator_rejects_missing_or_wrong_token(tmp_path):
    service = _service(tmp_path)
    server = create_server(("127.0.0.1", 0), service, token="team-secret", upload_dir=tmp_path / "uploads")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = CoordinatorClient(f"http://127.0.0.1:{server.server_port}", "wrong")
        with pytest.raises(RuntimeError, match="authentication failed"):
            client.snapshot()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_cli_join_persists_team_connection_for_later_commands(tmp_path):
    service = _service(tmp_path / "server")
    server = create_server(("127.0.0.1", 0), service, token="team-secret", upload_dir=tmp_path / "uploads")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "client" / "team.json"
    env = os.environ.copy()
    env["AUTOGGML_TEAM_CONFIG"] = str(config)
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        joined = subprocess.run(
            [
                sys.executable, "-m", "cli", "join", "--team", url, "--token", "team-secret",
                "--name", "friendly-box", "--machine-id", "box-id", "--backend", "cuda", "--memory-gib", "24",
            ],
            text=True, capture_output=True, env=env, check=False,
        )
        status = subprocess.run(
            [sys.executable, "-m", "cli", "status", "--json"],
            text=True, capture_output=True, env=env, check=False,
        )

        assert joined.returncode == 0, joined.stderr
        assert json.loads(status.stdout)["workers"][0]["name"] == "friendly-box"
        assert config.stat().st_mode & 0o077 == 0
        assert "team-secret" not in joined.stdout
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_worker_claims_runs_and_completes_only_its_assigned_job(tmp_path):
    patch = tmp_path / "candidate.patch"
    patch.write_text("experiment\n")
    service = _service(tmp_path)
    worker = service.join(JoinRequest("lucebox5", "machine-5", ["hip"], 128.0))
    service.submit(CandidateRequest("Experiment", patch, ["hip"], "deepseek-v4-flash"))
    calls = []

    result = TeamWorker(service, worker.worker_id, lambda claim: calls.append(claim) or {"verdict": "kept"}).run_once()

    assert result == {"status": "completed", "result": {"verdict": "kept"}}
    assert calls[0].candidate.title == "Experiment"
    assert calls[0].patch == patch.read_bytes()
    completed = service.snapshot().jobs[0]
    assert completed.status == "completed"
    assert completed.result == {"verdict": "kept"}
    assert service.snapshot().to_dict()["summary"]["ready"] == 1


def test_worker_records_executor_failure_and_releases_machine(tmp_path):
    patch = tmp_path / "candidate.patch"
    patch.write_text("experiment\n")
    service = _service(tmp_path)
    worker = service.join(JoinRequest("lucebox5", "machine-5", ["hip"], 128.0))
    service.submit(CandidateRequest("Experiment", patch, ["hip"], "deepseek-v4-flash"))

    def fail(_claim):
        raise RuntimeError("benchmark failed")

    result = TeamWorker(service, worker.worker_id, fail).run_once()

    assert result["status"] == "failed"
    assert "benchmark failed" in result["error"]
    assert service.snapshot().jobs[0].status == "failed"
    assert service.snapshot().to_dict()["summary"]["ready"] == 1
