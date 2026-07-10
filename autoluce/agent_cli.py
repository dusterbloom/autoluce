"""Agent-friendly CLI for choosing, executing, and crediting research tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from autoluce import ROOT
from autoluce.agent_challenges import (
    AgentJoinRequest,
    AgentOutput,
    AgentService,
    CandidatePatchGate,
    ChallengeRequest,
    FileAgentRepository,
)
from autoluce.agent_config import AgentIdentity
from autoluce.coordination import FileCoordinationRepository, FleetService
from autoluce.coordinator_http import AgentCoordinatorClient
from autoluce.parallel.worktree import ensure_worktree
from autoluce.source_layout import SourceLayout
from autoluce.team_config import TeamConnection


DEFAULT_COORDINATION_DIR = Path("~/.local/share/autoluce/team").expanduser()


def _gateway() -> AgentService | AgentCoordinatorClient:
    connection = TeamConnection.load()
    if connection:
        return AgentCoordinatorClient(connection.url, connection.token)
    root = Path(os.environ.get("AUTOLUCE_COORDINATION_DIR", DEFAULT_COORDINATION_DIR)).expanduser()
    fleet = FleetService(FileCoordinationRepository(root))
    return AgentService(FileAgentRepository(root / "agents"), fleet, CandidatePatchGate())


def _source_root() -> Path:
    return SourceLayout.resolve(root=ROOT).checkout


def _source_commit() -> str:
    override = os.environ.get("AUTOLUCE_SOURCE_COMMIT")
    if override:
        return override
    layout = SourceLayout.resolve(root=ROOT)
    pin = layout.pin_file
    if pin.exists():
        return pin.read_text().strip()
    source = layout.checkout
    if not source.exists():
        raise ValueError("Lucebox checkout is missing; run `uv run autoluce setup` before creating a challenge")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=True,
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoluce agent", description="Cooperate and compete on bounded research tasks")
    commands = parser.add_subparsers(dest="command", required=True)

    join = commands.add_parser("join", help="register an agent and its capabilities")
    join.add_argument("--name", required=True)
    join.add_argument("--capability", action="append", required=True, choices=["implement", "review", "recombine"])
    join.add_argument("--task-budget", type=int, default=10)
    join.add_argument("--json", action="store_true")

    challenge = commands.add_parser("challenge", help="create a fixed research challenge")
    challenge_commands = challenge.add_subparsers(dest="challenge_command", required=True)
    create = challenge_commands.add_parser("create")
    create.add_argument("--title", required=True)
    create.add_argument("--objective", required=True)
    create.add_argument("--why", required=True)
    create.add_argument("--evidence", action="append", default=[])
    create.add_argument("--model", required=True)
    create.add_argument("--backend", action="append", required=True, choices=["cuda", "hip"])
    create.add_argument("--slots", type=int, default=3)
    create.add_argument("--approach", action="append", default=[])
    create.add_argument("--token-budget", type=int, default=20_000)
    create.add_argument("--minutes", type=int, default=45)
    create.add_argument("--json", action="store_true")

    for name in ("next", "start"):
        command = commands.add_parser(name)
        if name == "start":
            command.add_argument("task_id")
            command.add_argument("--no-worktree", action="store_true")
        command.add_argument("--agent")
        command.add_argument("--json", action="store_true")

    submit = commands.add_parser("submit")
    submit.add_argument("task_id")
    submit.add_argument("--agent")
    submit.add_argument("--patch", type=Path)
    submit.add_argument("--rationale", required=True)
    submit.add_argument("--observation", action="append", default=[])
    submit.add_argument("--risk", action="append", default=[])
    submit.add_argument("--source", action="append", default=[])
    submit.add_argument("--json", action="store_true")

    advance = commands.add_parser("advance")
    advance.add_argument("challenge_id")
    advance.add_argument("--json", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")
    card = commands.add_parser("card")
    card.add_argument("challenge_id")
    card.add_argument("--json", action="store_true")
    return parser


def _emit(value: dict | list, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2))
    elif isinstance(value, list):
        if not value:
            print("No matching tasks are available.")
        for item in value:
            print(f"{item['task_id']}  {item['kind']}: {item['packet']['objective']}")
    else:
        print(json.dumps(value, indent=2))


def _agent_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    identity = AgentIdentity.load()
    if identity is None:
        raise ValueError("no default agent identity; run `autoluce agent join` or pass --agent")
    return identity.agent_id


def main() -> None:
    args = _parser().parse_args()
    gateway = _gateway()
    try:
        if args.command == "join":
            participant = gateway.join(AgentJoinRequest(args.name, args.capability, args.task_budget))
            AgentIdentity(participant.agent_id, participant.name).write()
            value = asdict(participant)
        elif args.command == "challenge":
            value = asdict(gateway.create_challenge(ChallengeRequest(
                args.title, args.objective, args.why, args.evidence, args.model, args.backend,
                base_commit=_source_commit(), implementation_slots=args.slots,
                token_budget=args.token_budget, time_budget_minutes=args.minutes,
                approaches=args.approach,
            )))
        elif args.command == "next":
            value = [asdict(task) for task in gateway.next_tasks(_agent_id(args.agent))]
        elif args.command == "start":
            agent_id = _agent_id(args.agent)
            task = gateway.claim(agent_id, args.task_id)
            context = gateway.context(agent_id, task.task_id)
            workspace = None
            if not args.no_worktree:
                path = ensure_worktree(_source_root(), f"agent-{task.task_id}", context.challenge.base_commit)
                metadata = path / ".autoluce" / "task.json"
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(json.dumps({
                    "agent_id": agent_id, "task": asdict(task), "challenge": asdict(context.challenge),
                }, indent=2) + "\n")
                workspace = str(path)
            value = {"task": asdict(task), "workspace": workspace}
        elif args.command == "submit":
            agent_id = _agent_id(args.agent)
            patch = args.patch.read_bytes() if args.patch else None
            artifact = gateway.finish(agent_id, args.task_id, AgentOutput(
                args.rationale, args.observation, args.risk, patch, args.source,
            ))
            value = {**asdict(artifact), "patch_path": str(artifact.patch_path) if artifact.patch_path else None}
        elif args.command == "advance":
            value = asdict(gateway.advance(args.challenge_id))
        elif args.command == "status":
            value = gateway.status()
        elif args.command == "card":
            value = gateway.challenge_card(args.challenge_id)
        else:
            raise ValueError(f"unsupported agent command: {args.command}")
        _emit(value, args.json)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"autoluce agent: {error}") from error


if __name__ == "__main__":
    main()
