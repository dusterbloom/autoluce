"""Read-only inventory of AutoLuce workspaces and their Git relationships.

Inventory is deliberately fail-closed: a legacy path has unknown ownership until a
later adoption step records it, and every unknown, dirty, locked, primary, external,
or non-Git path is ineligible for retirement.  This module never creates state and
sets ``GIT_OPTIONAL_LOCKS=0`` for every Git inspection so status checks do not refresh
the index as a side effect.
"""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from autoluce.workspace_registry import OwnershipRecord


SizeProvider = Callable[[Path], int]


@dataclass(frozen=True)
class WorkspaceEntry:
    name: str
    path: Path
    relative_path: str | None
    external: bool
    kind: str
    size_bytes: int
    size_known: bool
    ownership: str
    owner: str | None
    lease: str | None
    source_state: str
    git_common_dir: Path | None
    git_dir: Path | None
    head: str | None
    branch: str | None
    primary: bool
    busy: bool
    staged: int
    unstaged: int
    untracked: int
    conflicts: int
    retirement_blockers: tuple[str, ...]
    retirement_eligible: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "relative_path": self.relative_path,
            "external": self.external,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "size_known": self.size_known,
            "ownership": self.ownership,
            "owner": self.owner,
            "lease": self.lease,
            "source_state": self.source_state,
            "git_common_dir": str(self.git_common_dir) if self.git_common_dir else None,
            "git_dir": str(self.git_dir) if self.git_dir else None,
            "head": self.head,
            "branch": self.branch,
            "primary": self.primary,
            "busy": self.busy,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
            "conflicts": self.conflicts,
            "retirement_blockers": list(self.retirement_blockers),
            "retirement_eligible": self.retirement_eligible,
        }


@dataclass(frozen=True)
class GitWorkspaceGroup:
    common_dir: Path
    entries: tuple[WorkspaceEntry, ...]

    def to_dict(self) -> dict:
        return {
            "common_dir": str(self.common_dir),
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class WorkspaceInventory:
    work_root: Path
    git_groups: tuple[GitWorkspaceGroup, ...]
    ungrouped: tuple[WorkspaceEntry, ...]

    @property
    def entries(self) -> tuple[WorkspaceEntry, ...]:
        grouped = (entry for group in self.git_groups for entry in group.entries)
        return tuple(sorted((*grouped, *self.ungrouped), key=lambda entry: str(entry.path)))

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def total_size_bytes(self) -> int:
        # Size providers inspect each top-level tree recursively. Registered worktrees
        # nested inside another entry are useful context but are already included in
        # their ancestor's allocated bytes.
        return sum(
            entry.size_bytes
            for entry in self.entries
            if not entry.external and entry.relative_path and len(Path(entry.relative_path).parts) == 1
        )

    @property
    def external_size_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries if entry.external and entry.size_known)

    def to_dict(self) -> dict:
        entries = self.entries
        return {
            "schema_version": 1,
            "work_root": str(self.work_root),
            "summary": {
                "attested": sum(entry.ownership == "attested" for entry in entries),
                "entries": len(entries),
                "external_size_bytes": self.external_size_bytes,
                "external_size_unknown": sum(entry.external and not entry.size_known for entry in entries),
                "git_groups": len(self.git_groups),
                "managed": sum(entry.ownership == "managed" for entry in entries),
                "retirement_eligible": sum(entry.retirement_eligible for entry in entries),
                "size_bytes": self.total_size_bytes,
                "unknown": sum(entry.ownership == "unknown" for entry in entries),
            },
            "git_groups": [group.to_dict() for group in self.git_groups],
            "ungrouped": [entry.to_dict() for entry in self.ungrouped],
        }


@dataclass(frozen=True)
class _GitWorktreeRecord:
    path: Path
    head: str | None = None
    branch: str | None = None
    locked: bool = False
    prunable: bool = False
    bare: bool = False


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(path), *args],
        check=check,
        capture_output=True,
        env=env,
    )


def _git_path(path: Path, name: str) -> Path:
    value = os.fsdecode(_git(path, "rev-parse", "--path-format=absolute", name).stdout).strip()
    return Path(value).resolve(strict=False)


def _parse_worktree_list(payload: bytes) -> tuple[_GitWorktreeRecord, ...]:
    records: list[_GitWorktreeRecord] = []
    fields: dict[str, str | bool] = {}
    for raw in payload.split(b"\0"):
        if not raw:
            if fields:
                records.append(
                    _GitWorktreeRecord(
                        path=Path(str(fields["worktree"])).resolve(strict=False),
                        head=str(fields["HEAD"]) if "HEAD" in fields else None,
                        branch=str(fields["branch"]) if "branch" in fields else None,
                        locked=bool(fields.get("locked", False)),
                        prunable=bool(fields.get("prunable", False)),
                        bare=bool(fields.get("bare", False)),
                    )
                )
                fields = {}
            continue
        key, separator, value = os.fsdecode(raw).partition(" ")
        fields[key] = value if separator else True
    if fields:
        records.append(
            _GitWorktreeRecord(
                path=Path(str(fields["worktree"])).resolve(strict=False),
                head=str(fields["HEAD"]) if "HEAD" in fields else None,
                branch=str(fields["branch"]) if "branch" in fields else None,
                locked=bool(fields.get("locked", False)),
                prunable=bool(fields.get("prunable", False)),
                bare=bool(fields.get("bare", False)),
            )
        )
    return tuple(records)


def _source_counts(path: Path) -> tuple[str, int, int, int, int]:
    result = _git(
        path,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        check=False,
    )
    if result.returncode != 0:
        return "error", 0, 0, 0, 0
    staged = unstaged = untracked = conflicts = 0
    tokens = result.stdout.split(b"\0")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        record_type = token[:1]
        if record_type in (b"1", b"2"):
            fields = token.split(b" ", 2)
            xy = fields[1] if len(fields) > 1 else b".."
            staged += bool(xy[:1] and xy[:1] != b".")
            unstaged += bool(xy[1:2] and xy[1:2] != b".")
            if record_type == b"2":
                index += 1  # porcelain v2 emits the rename/copy source as another NUL field
        elif record_type == b"u":
            conflicts += 1
        elif record_type == b"?":
            untracked += 1
    dirty = staged + unstaged + untracked + conflicts
    return ("dirty" if dirty else "clean"), staged, unstaged, untracked, conflicts


def _has_git_lock(git_dir: Path | None, common_dir: Path) -> bool:
    candidates: set[Path] = {
        common_dir / "config.lock",
        common_dir / "packed-refs.lock",
        common_dir / "shallow.lock",
    }
    if git_dir:
        candidates.update({git_dir / "index.lock", git_dir / "HEAD.lock"})
    if any(path.exists() for path in candidates):
        return True
    refs = common_dir / "refs"
    return refs.is_dir() and any(refs.rglob("*.lock"))


def _has_git_operation(git_dir: Path | None) -> bool:
    if not git_dir:
        return False
    markers = (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_START",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
    )
    return any((git_dir / marker).exists() for marker in markers)


def _relationship(path: Path, work_root: Path) -> tuple[bool, str | None]:
    try:
        return False, str(path.relative_to(work_root))
    except ValueError:
        return True, None


def _entry_size(path: Path, external: bool, work_root: Path, size_provider: SizeProvider) -> tuple[int, bool]:
    if not path.exists() and not path.is_symlink():
        return 0, False
    # An external primary can be an ancestor of work_root; recursively sizing it
    # would count the managed tree again and may traverse an entire user checkout.
    if external and path in work_root.parents:
        return 0, False
    try:
        size = int(size_provider(path))
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0, False
    return max(0, size), True


def _ownership(path: Path, records: Mapping[str, OwnershipRecord]) -> tuple[str, str | None]:
    record = records.get(str(path))
    if not record:
        return "unknown", None
    return record.custody, record.owner


def _git_entry(
    record: _GitWorktreeRecord,
    common_dir: Path,
    work_root: Path,
    size_provider: SizeProvider,
    ownership_records: Mapping[str, OwnershipRecord],
) -> WorkspaceEntry:
    path = record.path
    external, relative_path = _relationship(path, work_root)
    exists = path.is_dir()
    git_dir: Path | None = None
    source_state = "missing"
    staged = unstaged = untracked = conflicts = 0
    if exists:
        try:
            git_dir = _git_path(path, "--git-dir")
            if record.bare:
                source_state = "not-applicable"
            else:
                source_state, staged, unstaged, untracked, conflicts = _source_counts(path)
        except subprocess.CalledProcessError:
            source_state = "error"
    primary = git_dir == common_dir if git_dir else False
    lock_present = record.locked or _has_git_lock(git_dir, common_dir)
    operation_active = _has_git_operation(git_dir)
    busy = lock_present or operation_active
    size_bytes, size_known = _entry_size(path, external, work_root, size_provider)
    ownership, owner = _ownership(path, ownership_records)

    blockers: list[str] = []
    if ownership == "unknown":
        blockers.append("ownership-unknown")
    elif ownership == "attested":
        blockers.append("not-managed")
    else:
        blockers.append("lifecycle-not-sealed")
    if primary:
        blockers.append("primary-worktree")
    if external:
        blockers.append("outside-managed-root")
    if not exists or record.prunable:
        blockers.append("missing-worktree")
    if source_state == "dirty":
        blockers.append("dirty-source")
    elif source_state == "error":
        blockers.append("status-unavailable")
    if lock_present:
        blockers.append("git-lock-present")
    if operation_active:
        blockers.append("git-operation-active")
    if not size_known and not external and exists:
        blockers.append("size-unavailable")

    return WorkspaceEntry(
        name=path.name or str(path),
        path=path,
        relative_path=relative_path,
        external=external,
        kind="git-bare" if record.bare else "git-worktree",
        size_bytes=size_bytes,
        size_known=size_known,
        ownership=ownership,
        owner=owner,
        lease=None,
        source_state=source_state,
        git_common_dir=common_dir,
        git_dir=git_dir,
        head=record.head,
        branch=record.branch,
        primary=primary,
        busy=busy,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        conflicts=conflicts,
        retirement_blockers=tuple(blockers),
        retirement_eligible=False,
    )


def _ungrouped_entry(
    path: Path,
    work_root: Path,
    size_provider: SizeProvider,
    ownership_records: Mapping[str, OwnershipRecord],
) -> WorkspaceEntry:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        kind, reason = "symlink", "symlink"
    elif stat.S_ISDIR(mode):
        kind, reason = "directory", "not-git-worktree"
    elif stat.S_ISREG(mode):
        kind, reason = "file", "not-git-worktree"
    else:
        kind, reason = "other", "unsupported-file-type"
    size_bytes, size_known = _entry_size(path, False, work_root, size_provider)
    entry_path = path.resolve(strict=False) if kind != "symlink" else path.absolute()
    ownership, owner = _ownership(entry_path, ownership_records)
    blockers = ["ownership-unknown" if ownership == "unknown" else "not-managed", reason]
    if not size_known:
        blockers.append("size-unavailable")
    return WorkspaceEntry(
        name=path.name,
        path=entry_path,
        relative_path=str(path.relative_to(work_root)),
        external=False,
        kind=kind,
        size_bytes=size_bytes,
        size_known=size_known,
        ownership=ownership,
        owner=owner,
        lease=None,
        source_state="not-git",
        git_common_dir=None,
        git_dir=None,
        head=None,
        branch=None,
        primary=False,
        busy=False,
        staged=0,
        unstaged=0,
        untracked=0,
        conflicts=0,
        retirement_blockers=tuple(blockers),
        retirement_eligible=False,
    )


def disk_usage(path: Path) -> int:
    """Return allocated bytes without following a top-level symlink."""
    if path.is_symlink() or not path.is_dir():
        return path.lstat().st_size
    result = subprocess.run(["du", "-sk", "--", str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0]) * 1024


def inventory_workspaces(
    work_root: Path,
    size_provider: SizeProvider = disk_usage,
    ownership_records: Mapping[str, OwnershipRecord] | None = None,
) -> WorkspaceInventory:
    """Inspect top-level work paths and every linked worktree sharing their Git stores."""
    work_root = work_root.resolve(strict=False)
    ownership_records = ownership_records or {}
    children = tuple(sorted(work_root.iterdir(), key=lambda path: path.name))
    seeds: list[Path] = []
    ungrouped: list[WorkspaceEntry] = []
    for child in children:
        if child.is_symlink():
            ungrouped.append(_ungrouped_entry(child, work_root, size_provider, ownership_records))
            continue
        git_worktree = (child / ".git").is_dir() or (child / ".git").is_file()
        bare_repository = (child / "HEAD").is_file() and (child / "objects").is_dir() and (child / "refs").is_dir()
        if child.is_dir() and (git_worktree or bare_repository):
            seeds.append(child)
        else:
            ungrouped.append(_ungrouped_entry(child, work_root, size_provider, ownership_records))

    groups: dict[Path, GitWorkspaceGroup] = {}
    for seed in seeds:
        try:
            common_dir = _git_path(seed, "--git-common-dir")
        except subprocess.CalledProcessError:
            ungrouped.append(_ungrouped_entry(seed, work_root, size_provider, ownership_records))
            continue
        if common_dir in groups:
            continue
        records = _parse_worktree_list(_git(seed, "worktree", "list", "--porcelain", "-z").stdout)
        entries = tuple(
            sorted(
                (_git_entry(record, common_dir, work_root, size_provider, ownership_records) for record in records),
                key=lambda entry: str(entry.path),
            )
        )
        groups[common_dir] = GitWorkspaceGroup(common_dir=common_dir, entries=entries)

    return WorkspaceInventory(
        work_root=work_root,
        git_groups=tuple(groups[key] for key in sorted(groups, key=str)),
        ungrouped=tuple(sorted(ungrouped, key=lambda entry: str(entry.path))),
    )
