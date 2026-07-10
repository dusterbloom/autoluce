"""Friendly CLI adapter for team coordination."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

from autoggml.coordination import CandidateRequest, FileCoordinationRepository, FleetService, JoinRequest
from autoggml.coordinator_http import CoordinatorClient
from autoggml.team_config import TeamConnection
from autoggml.team_worker import LocalExperimentExecutor, TeamWorker


DEFAULT_COORDINATION_DIR = Path("~/.local/share/autoggml/team").expanduser()


def _service() -> FleetService | CoordinatorClient:
    connection = TeamConnection.load()
    if connection:
        return CoordinatorClient(connection.url, connection.token)
    root = Path(os.environ.get("AUTOGGML_COORDINATION_DIR", DEFAULT_COORDINATION_DIR)).expanduser()
    return FleetService(FileCoordinationRepository(root))


def _machine_id() -> str:
    override = os.environ.get("AUTOGGML_MACHINE_ID")
    if override:
        return override
    path = Path("/etc/machine-id")
    return path.read_text().strip() if path.exists() else platform.node()


def _memory_gib() -> float:
    try:
        line = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
        return round(int(line.split()[1]) / 1024**2, 1)
    except (OSError, StopIteration, ValueError):
        return 1.0


def _backends() -> list[str]:
    found = []
    if shutil.which("nvcc") or Path("/dev/nvidia0").exists():
        found.append("cuda")
    if shutil.which("hipcc") or Path("/dev/kfd").exists():
        found.append("hip")
    return found


def _worker_id(service: FleetService | CoordinatorClient, explicit: str | None) -> str:
    if explicit:
        return explicit
    machine_id = _machine_id()
    matches = [worker.worker_id for worker in service.snapshot().workers if worker.machine_id == machine_id]
    if not matches:
        raise ValueError("this machine has not joined; run `autoggml join`")
    return matches[0]


def _parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"autoggml {action}")
    if action == "join":
        parser.add_argument("--name", default=platform.node())
        parser.add_argument("--machine-id", default=_machine_id())
        parser.add_argument("--backend", action="append", choices=["cuda", "hip"])
        parser.add_argument("--memory-gib", type=float, default=_memory_gib())
        parser.add_argument("--team", help="shared coordinator URL (remembered after joining)")
        parser.add_argument("--token", help="shared team token (stored with user-only permissions)")
    elif action == "submit":
        parser.add_argument("patch", type=Path)
        parser.add_argument("--title")
        parser.add_argument("--backend", action="append", required=True, choices=["cuda", "hip"])
        parser.add_argument("--model", required=True)
    elif action == "status":
        parser.add_argument("--json", action="store_true")
    elif action == "worker":
        parser.add_argument("--worker")
        parser.add_argument("--once", action="store_true", help="process one assigned experiment and exit")
        parser.add_argument("--simulate", action="store_true", help="exercise the lifecycle without accelerator work")
        parser.add_argument("--lock-path", default="/tmp/autoggml-gpu.lock")
    else:
        parser.add_argument("--worker")
    return parser


def main(action: str | None = None, argv: list[str] | None = None) -> None:
    if action is None:
        raise SystemExit("fleet action was not supplied")
    args = _parser(action).parse_args(argv)
    try:
        if action == "join" and (args.team or args.token):
            if not args.team or not args.token:
                raise ValueError("--team and --token must be supplied together")
            TeamConnection(args.team, args.token).write()
        service = _service()
        if action == "join":
            backends = args.backend or _backends()
            worker = service.join(JoinRequest(args.name, args.machine_id, backends, args.memory_gib))
            print(f"{worker.name} is ready ({', '.join(worker.backends)}, {worker.memory_gib:g} GiB)")
            print("Next: autoggml status")
        elif action == "submit":
            request = CandidateRequest(args.title or args.patch.stem, args.patch, args.backend, args.model)
            job = service.submit(request)
            worker = next(worker for worker in service.snapshot().workers if worker.worker_id == job.worker_id)
            print(f"Queued on {worker.name}: {job.job_id}")
            print("Watch: autoggml status")
        elif action == "status":
            payload = service.snapshot().to_dict()
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                summary = payload["summary"]
                print(f"Team: {summary['machines']} machines, {summary['ready']} ready, {summary['queued']} queued, {summary['running']} running")
                for worker in payload["workers"]:
                    print(f"  {worker['name']}: {worker['state']} ({', '.join(worker['backends'])})")
                for job in payload["jobs"]:
                    print(f"  {job['title']}: {job['status']} on {job['worker']}")
        elif action == "worker":
            if not args.once:
                raise ValueError("continuous workers are not installed yet; use --once")
            worker_id = _worker_id(service, args.worker)
            result = TeamWorker(
                service, worker_id, LocalExperimentExecutor(simulate=args.simulate, lock_path=args.lock_path),
            ).run_once()
            if result["status"] == "idle":
                print("No experiment is assigned to this machine.")
            elif result["status"] == "completed":
                print("Experiment completed and the result was returned to the team.")
            else:
                raise RuntimeError(result["error"])
        elif action in {"pause", "resume"}:
            worker_id = _worker_id(service, args.worker)
            worker = service.pause(worker_id) if action == "pause" else service.resume(worker_id)
            print(f"{worker.name} is {worker.state}")
        elif action == "leave":
            worker_id = _worker_id(service, args.worker)
            name = next(worker.name for worker in service.snapshot().workers if worker.worker_id == worker_id)
            service.leave(worker_id)
            print(f"{name} left the team")
        else:
            raise ValueError(f"unsupported fleet action: {action}")
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"autoggml: {error}") from error


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2:])
