"""Workspace lifecycle inspection commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoluce import ROOT
from autoluce.workspace_inventory import SizeProvider, disk_usage, inventory_workspaces
from autoluce.workspace_registry import WorkspaceRegistry


DEFAULT_WORKSPACE_STATE = Path("~/.local/share/autoluce/workspaces").expanduser()


def _work_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("AUTOLUCE_WORK_ROOT", ROOT / "work")),
        help="managed work directory (default: AUTOLUCE_WORK_ROOT or repository work/)",
    )


def _state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("AUTOLUCE_WORKSPACE_STATE", DEFAULT_WORKSPACE_STATE)),
        help="local workspace registry (default: AUTOLUCE_WORKSPACE_STATE or ~/.local/share/autoluce/workspaces)",
    )


def _human_size(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None, size_provider: SizeProvider | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoluce workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="inventory workspaces without changing them")
    status.add_argument("--json", action="store_true")
    _work_root_argument(status)
    _state_argument(status)
    attest = commands.add_parser("attest", help="record who owns legacy paths without granting deletion authority")
    attest.add_argument("--all", action="store_true", required=True, help="attest every path reported by status")
    attest.add_argument("--owner", required=True, help="stable person or team identity")
    attest.add_argument("--apply", action="store_true", help="write the registry; the default is a dry run")
    _work_root_argument(attest)
    _state_argument(attest)
    args = parser.parse_args(argv)

    registry = WorkspaceRegistry(args.state_dir)
    records = registry.load()
    inventory = inventory_workspaces(
        args.work_root,
        size_provider=size_provider or disk_usage,
        ownership_records=records,
    )
    if args.command == "attest":
        entries = inventory.entries
        mode = "applied" if args.apply else "dry-run"
        if args.apply:
            registry.attest(entries, owner=args.owner)
        print(
            json.dumps(
                {"mode": mode, "owner": args.owner, "paths": [str(entry.path) for entry in entries]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.json:
        print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return 0

    summary = inventory.to_dict()["summary"]
    print(f"Workspace inventory: {summary['entries']} paths in {summary['git_groups']} Git groups")
    print(f"Managed disk usage: {_human_size(summary['size_bytes'])}")
    if summary["external_size_bytes"] or summary["external_size_unknown"]:
        print(
            f"External worktree usage: {_human_size(summary['external_size_bytes'])}; "
            f"unknown sizes: {summary['external_size_unknown']}"
        )
    print(
        f"Ownership: {summary['managed']} managed, {summary['attested']} attested, "
        f"{summary['unknown']} unknown; "
        f"retirement eligible: {summary['retirement_eligible']}"
    )
    for entry in inventory.entries:
        blockers = ", ".join(entry.retirement_blockers) or "none"
        location = f"external:{entry.path}" if entry.external else entry.relative_path
        print(f"  {location} [{entry.kind}, {entry.source_state}] blockers: {blockers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
