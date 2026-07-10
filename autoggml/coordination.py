"""Small, deterministic coordination core for a distributed autoggml team.

The service owns policy. Repositories own persistence. CLI and future HTTP/GitHub
adapters can therefore share routing rules without duplicating them.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar


from autoggml.atomic_store import AtomicJsonStore
from autoggml.identifiers import stable_id
ACTIVE_JOB_STATES = {"queued", "running"}
T = TypeVar("T")


class NoCompatibleWorkerError(RuntimeError):
    """No currently available worker satisfies a candidate request."""


@dataclass(frozen=True)
class JoinRequest:
    name: str
    machine_id: str
    backends: list[str]
    memory_gib: float


@dataclass(frozen=True)
class CandidateRequest:
    title: str
    patch_path: Path
    backends: list[str]
    model: str


@dataclass(frozen=True)
class Worker:
    worker_id: str
    name: str
    machine_id: str
    backends: list[str]
    memory_gib: float
    state: str = "ready"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    title: str
    model: str
    backends: list[str]
    patch_sha256: str
    patch_path: Path


@dataclass(frozen=True)
class Job:
    job_id: str
    candidate_id: str
    worker_id: str
    status: str = "queued"
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class Claim:
    job: Job
    candidate: Candidate
    patch: bytes


@dataclass(frozen=True)
class FleetSnapshot:
    workers: list[Worker]
    candidates: list[Candidate]
    jobs: list[Job]

    def to_dict(self) -> dict:
        candidates = {candidate.candidate_id: candidate for candidate in self.candidates}
        workers = {worker.worker_id: worker for worker in self.workers}
        worker_rows = [asdict(worker) for worker in self.workers]
        candidate_rows = [{**asdict(candidate), "patch_path": str(candidate.patch_path)} for candidate in self.candidates]
        job_rows = []
        for job in self.jobs:
            candidate = candidates[job.candidate_id]
            worker = workers[job.worker_id]
            job_rows.append({**asdict(job), "title": candidate.title, "model": candidate.model, "worker": worker.name})
        active = {job.worker_id for job in self.jobs if job.status in ACTIVE_JOB_STATES}
        return {
            "summary": {
                "machines": len(self.workers),
                "ready": sum(worker.state == "ready" and worker.worker_id not in active for worker in self.workers),
                "running": sum(job.status == "running" for job in self.jobs),
                "queued": sum(job.status == "queued" for job in self.jobs),
            },
            "workers": worker_rows,
            "candidates": candidate_rows,
            "jobs": job_rows,
        }


class CoordinationRepository(Protocol):
    @property
    def candidates_dir(self) -> Path: ...

    def read(self) -> FleetSnapshot: ...

    def update(self, operation: Callable[[dict], T]) -> T: ...


class FileCoordinationRepository:
    """Atomic, process-safe repository suitable for one coordinator host."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.state_path = self.root / "state.json"
        self.store = AtomicJsonStore(
            self.state_path,
            lambda: {"workers": [], "candidates": [], "jobs": [], "schema_version": 1},
        )

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    def _ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(exist_ok=True)

    def update(self, operation: Callable[[dict], T]) -> T:
        self._ensure()
        return self.store.update(operation)

    def read(self) -> FleetSnapshot:
        def load(state: dict) -> FleetSnapshot:
            return _snapshot(state)

        return self.update(load)


def _snapshot(state: dict) -> FleetSnapshot:
    return FleetSnapshot(
        workers=[Worker(**value) for value in state["workers"]],
        candidates=[Candidate(**{**value, "patch_path": Path(value["patch_path"])}) for value in state["candidates"]],
        jobs=[Job(**value) for value in state["jobs"]],
    )


class FleetService:
    def __init__(self, repository: CoordinationRepository) -> None:
        self.repository = repository

    def join(self, request: JoinRequest) -> Worker:
        if not request.name.strip() or not request.machine_id.strip():
            raise ValueError("machine name and identity are required")
        backends = sorted(set(request.backends))
        if not backends:
            raise ValueError("at least one backend is required")
        if request.memory_gib <= 0:
            raise ValueError("memory must be positive")
        worker_id = stable_id("worker", request.machine_id)

        def apply(state: dict) -> Worker:
            worker = Worker(worker_id, request.name.strip(), request.machine_id, backends, request.memory_gib)
            for index, existing in enumerate(state["workers"]):
                if existing["machine_id"] == request.machine_id:
                    worker = Worker(
                        existing["worker_id"], request.name.strip(), request.machine_id, backends,
                        request.memory_gib, existing["state"],
                    )
                    state["workers"][index] = asdict(worker)
                    return worker
            state["workers"].append(asdict(worker))
            return worker

        return self.repository.update(apply)

    def submit(self, request: CandidateRequest) -> Job:
        patch_path = request.patch_path.expanduser().resolve()
        if not patch_path.is_file():
            raise ValueError(f"patch does not exist: {patch_path}")
        backends = sorted(set(request.backends))
        if not backends:
            raise ValueError("at least one backend is required")
        patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        candidate_id = stable_id("candidate", patch_sha, request.model, ",".join(backends))
        stored_patch = self.repository.candidates_dir / f"{candidate_id}.patch"

        def apply(state: dict) -> Job:
            for job_value in state["jobs"]:
                if job_value["candidate_id"] == candidate_id:
                    return Job(**job_value)

            snapshot = _snapshot(state)
            occupied = {job.worker_id for job in snapshot.jobs if job.status in ACTIVE_JOB_STATES}
            eligible = [
                worker for worker in snapshot.workers
                if worker.state == "ready"
                and worker.worker_id not in occupied
                and set(backends).issubset(worker.backends)
            ]
            if not eligible:
                label = "+".join(backends)
                raise NoCompatibleWorkerError(f"No idle machine supports {label}. Try `autoggml status` or resume a machine.")
            worker = sorted(eligible, key=lambda item: (-item.memory_gib, item.name, item.worker_id))[0]

            if not any(value["candidate_id"] == candidate_id for value in state["candidates"]):
                self.repository.candidates_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(patch_path, stored_patch)
                candidate = Candidate(
                    candidate_id, request.title.strip() or patch_path.stem, request.model, backends, patch_sha, stored_patch,
                )
                state["candidates"].append({**asdict(candidate), "patch_path": str(stored_patch)})
            job = Job(stable_id("job", candidate_id, worker.worker_id), candidate_id, worker.worker_id)
            state["jobs"].append(asdict(job))
            return job

        return self.repository.update(apply)

    def pause(self, worker_id: str) -> Worker:
        return self._set_state(worker_id, "paused")

    def resume(self, worker_id: str) -> Worker:
        return self._set_state(worker_id, "ready")

    def _set_state(self, worker_id: str, desired: str) -> Worker:
        def apply(state: dict) -> Worker:
            for index, value in enumerate(state["workers"]):
                if value["worker_id"] == worker_id:
                    worker = Worker(**{**value, "state": desired})
                    state["workers"][index] = asdict(worker)
                    return worker
            raise ValueError(f"unknown worker: {worker_id}")

        return self.repository.update(apply)

    def leave(self, worker_id: str) -> None:
        def apply(state: dict) -> None:
            if any(job["worker_id"] == worker_id and job["status"] in ACTIVE_JOB_STATES for job in state["jobs"]):
                raise ValueError("machine has assigned work; finish or cancel it before leaving")
            before = len(state["workers"])
            state["workers"] = [worker for worker in state["workers"] if worker["worker_id"] != worker_id]
            if len(state["workers"]) == before:
                raise ValueError(f"unknown worker: {worker_id}")

        self.repository.update(apply)

    def claim(self, worker_id: str) -> Claim | None:
        def apply(state: dict) -> Claim | None:
            workers = {worker["worker_id"] for worker in state["workers"]}
            if worker_id not in workers:
                raise ValueError(f"unknown worker: {worker_id}")
            for index, value in enumerate(state["jobs"]):
                if value["worker_id"] != worker_id or value["status"] != "queued":
                    continue
                job = Job(**{**value, "status": "running"})
                state["jobs"][index] = asdict(job)
                candidate_value = next(
                    candidate for candidate in state["candidates"] if candidate["candidate_id"] == job.candidate_id
                )
                candidate = Candidate(**{**candidate_value, "patch_path": Path(candidate_value["patch_path"])})
                return Claim(job, candidate, candidate.patch_path.read_bytes())
            return None

        return self.repository.update(apply)

    def finish(self, job_id: str, status: str, result: dict[str, Any]) -> Job:
        if status not in {"completed", "failed"}:
            raise ValueError("finished job status must be completed or failed")

        def apply(state: dict) -> Job:
            for index, value in enumerate(state["jobs"]):
                if value["job_id"] == job_id:
                    if value["status"] != "running":
                        raise ValueError(f"job is not running: {job_id}")
                    job = Job(**{**value, "status": status, "result": result})
                    state["jobs"][index] = asdict(job)
                    return job
            raise ValueError(f"unknown job: {job_id}")

        return self.repository.update(apply)

    def snapshot(self) -> FleetSnapshot:
        return self.repository.read()
