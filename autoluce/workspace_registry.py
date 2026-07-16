"""Local ownership attestations for workspace paths.

An attestation answers who owns a legacy path.  It intentionally does *not* grant
AutoLuce custody or deletion authority; managed custody requires the later nonce,
lease, and snapshot protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from autoluce.atomic_store import AtomicJsonStore

if TYPE_CHECKING:
    from autoluce.workspace_inventory import WorkspaceEntry


@dataclass(frozen=True)
class OwnershipRecord:
    path: str
    kind: str
    owner: str
    custody: str
    git_common_dir: str | None
    git_dir: str | None
    observed_head: str | None
    attested_at: str


def _default_state() -> dict:
    return {"schema_version": 1, "workspaces": []}


class WorkspaceRegistry:
    """Atomic local registry stored outside the source worktree."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.path = self.root / "state.json"
        self.store = AtomicJsonStore(self.path, _default_state)

    def load(self) -> dict[str, OwnershipRecord]:
        """Read records without creating the registry or taking a write lock."""
        if not self.path.exists():
            return {}
        import json

        state = json.loads(self.path.read_text())
        if state.get("schema_version") != 1:
            raise ValueError(f"unsupported workspace registry schema in {self.path}")
        records = [OwnershipRecord(**value) for value in state.get("workspaces", [])]
        if any(record.custody != "attested" for record in records):
            raise ValueError("workspace registry schema 1 attests ownership but does not grant managed custody")
        paths = [record.path for record in records]
        if len(paths) != len(set(paths)):
            raise ValueError("workspace registry contains duplicate paths")
        return {record.path: record for record in records}

    def attest(self, entries: Iterable[WorkspaceEntry], owner: str) -> tuple[OwnershipRecord, ...]:
        owner = owner.strip()
        if not owner:
            raise ValueError("owner must not be empty")
        entries = tuple(entries)

        def operation(state: dict) -> tuple[OwnershipRecord, ...]:
            if state.get("schema_version") != 1:
                raise ValueError("unsupported workspace registry schema")
            by_path = {value["path"]: value for value in state["workspaces"]}
            results: list[OwnershipRecord] = []
            now = datetime.now(timezone.utc).isoformat()
            for entry in entries:
                path = str(entry.path)
                existing = by_path.get(path)
                if existing:
                    if existing["owner"] != owner:
                        raise ValueError(f"{path} is already attested to {existing['owner']}")
                    results.append(OwnershipRecord(**existing))
                    continue
                record = OwnershipRecord(
                    path=path,
                    kind=entry.kind,
                    owner=owner,
                    custody="attested",
                    git_common_dir=str(entry.git_common_dir) if entry.git_common_dir else None,
                    git_dir=str(entry.git_dir) if entry.git_dir else None,
                    observed_head=entry.head,
                    attested_at=now,
                )
                value = asdict(record)
                state["workspaces"].append(value)
                by_path[path] = value
                results.append(record)
            state["workspaces"].sort(key=lambda value: value["path"])
            return tuple(results)

        return self.store.update(operation)
