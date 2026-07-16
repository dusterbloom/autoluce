from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from autoluce.workspace_inventory import inventory_workspaces
from autoluce.workspace_cli import main
from autoluce.workspace_registry import WorkspaceRegistry


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@autoluce.local")
    _git(path, "config", "user.name", "AutoLuce Test")
    (path / "tracked.cpp").write_text("base\n")
    _git(path, "add", "tracked.cpp")
    _git(path, "commit", "-qm", "base")
    return path


def _entry(inventory, name: str):
    matches = [entry for entry in inventory.entries if entry.name == name]
    assert len(matches) == 1
    return matches[0]


def test_inventory_groups_linked_worktrees_and_fails_closed(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    linked = work_root / "linked"
    external = tmp_path / "external"
    _git(primary, "worktree", "add", "--detach", str(linked), "HEAD")
    _git(primary, "worktree", "add", "--detach", str(external), "HEAD")
    (linked / "tracked.cpp").write_text("dirty\n")
    build = work_root / "build-cache"
    build.mkdir()
    (build / "artifact.bin").write_bytes(b"artifact")

    inventory = inventory_workspaces(work_root, size_provider=lambda path: 100 + len(path.name))

    assert inventory.work_root == work_root.resolve()
    assert inventory.total_entries == 4
    assert inventory.total_size_bytes == sum(
        entry.size_bytes for entry in inventory.entries if not entry.external
    )
    assert len(inventory.git_groups) == 1
    assert {entry.name for entry in inventory.git_groups[0].entries} == {
        "primary",
        "linked",
        "external",
    }

    primary_entry = _entry(inventory, "primary")
    assert primary_entry.kind == "git-worktree"
    assert primary_entry.primary
    assert primary_entry.source_state == "clean"
    assert primary_entry.ownership == "unknown"
    assert primary_entry.lease is None
    assert primary_entry.retirement_blockers == (
        "ownership-unknown",
        "primary-worktree",
    )

    linked_entry = _entry(inventory, "linked")
    assert linked_entry.source_state == "dirty"
    assert linked_entry.unstaged == 1
    assert "dirty-source" in linked_entry.retirement_blockers
    assert not linked_entry.external

    external_entry = _entry(inventory, "external")
    assert external_entry.external
    assert external_entry.relative_path is None
    assert "outside-managed-root" in external_entry.retirement_blockers

    build_entry = _entry(inventory, "build-cache")
    assert build_entry.kind == "directory"
    assert build_entry.source_state == "not-git"
    assert build_entry.git_common_dir is None
    assert build_entry.retirement_blockers == (
        "ownership-unknown",
        "not-git-worktree",
    )


def test_inventory_does_not_refresh_git_index_or_create_state(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    index = Path(_git(primary, "rev-parse", "--git-path", "index"))
    if not index.is_absolute():
        index = primary / index
    before_bytes = index.read_bytes()
    before_stat = index.stat()
    before_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    inventory_workspaces(work_root, size_provider=lambda _path: 0)

    after_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    after_stat = index.stat()
    assert index.read_bytes() == before_bytes
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_paths == before_paths


def test_workspace_status_json_is_deterministic_and_read_only(tmp_path, monkeypatch, capsys):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _make_repository(work_root / "repo")
    monkeypatch.setenv("AUTOLUCE_WORK_ROOT", str(work_root))

    assert main(["status", "--json"], size_provider=lambda _path: 7) == 0
    first = capsys.readouterr().out
    assert main(["status", "--json"], size_provider=lambda _path: 7) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["work_root"] == str(work_root.resolve())
    assert payload["summary"] == {
        "attested": 0,
        "entries": 1,
        "external_size_bytes": 0,
        "external_size_unknown": 0,
        "git_groups": 1,
        "managed": 0,
        "retirement_eligible": 0,
        "size_bytes": 7,
        "unknown": 1,
    }
    assert payload["git_groups"][0]["entries"][0]["ownership"] == "unknown"


def test_inventory_marks_git_lock_as_busy_without_removing_it(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    lock = Path(_git(primary, "rev-parse", "--git-path", "index.lock"))
    if not lock.is_absolute():
        lock = primary / lock
    lock.write_text("busy")

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 0)

    entry = _entry(inventory, "primary")
    assert entry.busy
    assert "git-lock-present" in entry.retirement_blockers
    assert lock.read_text() == "busy"


def test_inventory_ignores_symlink_targets(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    target = tmp_path / "model-weights"
    target.mkdir()
    (target / "model.gguf").write_bytes(b"weights")
    os.symlink(target, work_root / "models")

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 3)

    entry = _entry(inventory, "models")
    assert entry.kind == "symlink"
    assert entry.size_bytes == 3
    assert entry.retirement_blockers == (
        "ownership-unknown",
        "symlink",
    )


def test_attested_ownership_is_distinct_from_managed_custody(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _make_repository(work_root / "repo")
    build = work_root / "build-cache"
    build.mkdir()
    state_dir = tmp_path / "state"
    registry = WorkspaceRegistry(state_dir)

    before = inventory_workspaces(work_root, size_provider=lambda _path: 0)
    registry.attest(before.entries, owner="peppi")
    after = inventory_workspaces(
        work_root,
        size_provider=lambda _path: 0,
        ownership_records=registry.load(),
    )

    for entry in after.entries:
        assert entry.owner == "peppi"
        assert entry.ownership == "attested"
        assert not entry.retirement_eligible
        assert "ownership-unknown" not in entry.retirement_blockers
        assert "not-managed" in entry.retirement_blockers


def test_workspace_attest_is_dry_run_until_apply(tmp_path, monkeypatch, capsys):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _make_repository(work_root / "repo")
    state_dir = tmp_path / "state"
    monkeypatch.setenv("AUTOLUCE_WORK_ROOT", str(work_root))
    monkeypatch.setenv("AUTOLUCE_WORKSPACE_STATE", str(state_dir))

    assert main(["attest", "--all", "--owner", "peppi"], size_provider=lambda _path: 0) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "dry-run"
    assert dry_run["paths"] == [str((work_root / "repo").resolve())]
    assert not state_dir.exists()

    assert main(["attest", "--all", "--owner", "peppi", "--apply"], size_provider=lambda _path: 0) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["mode"] == "applied"
    records = WorkspaceRegistry(state_dir).load()
    assert records[str((work_root / "repo").resolve())].owner == "peppi"
    assert records[str((work_root / "repo").resolve())].custody == "attested"


def test_attestation_is_idempotent_and_rejects_owner_reassignment(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _make_repository(work_root / "repo")
    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 0)
    registry = WorkspaceRegistry(tmp_path / "state")

    registry.attest(inventory.entries, owner="peppi")
    registry.attest(inventory.entries, owner="peppi")
    assert len(registry.load()) == 1

    with pytest.raises(ValueError, match="already attested to peppi"):
        registry.attest(inventory.entries, owner="someone-else")


def test_inventory_disables_configured_fsmonitor_hook(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    marker = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor-hook"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    hook.chmod(0o755)
    _git(primary, "config", "core.fsmonitor", str(hook))

    inventory_workspaces(work_root, size_provider=lambda _path: 0)

    assert not marker.exists()


def test_inventory_does_not_honor_config_that_hides_dirty_submodule(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    child = _make_repository(tmp_path / "child")
    primary = _make_repository(work_root / "primary")
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child), "submodule"],
        cwd=primary,
        check=True,
    )
    _git(primary, "commit", "-qam", "add submodule")
    _git(primary, "config", "submodule.submodule.ignore", "all")
    (primary / "submodule" / "tracked.cpp").write_text("dirty\n")

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 0)

    entry = _entry(inventory, "primary")
    assert entry.source_state == "dirty"
    assert entry.unstaged == 1


def test_inventory_counts_nested_registered_worktree_bytes_once(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    nested = primary / "nested"
    _git(primary, "worktree", "add", "--detach", str(nested), "HEAD")
    sizes = {primary.resolve(): 1000, nested.resolve(): 200}

    inventory = inventory_workspaces(work_root, size_provider=lambda path: sizes[path.resolve()])

    assert inventory.total_entries == 2
    assert inventory.total_size_bytes == 1000


def test_inventory_recognizes_bare_git_repository(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    bare = work_root / "relay.git"
    bare.mkdir()
    _git(bare, "init", "--bare", "-q")

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 9)

    entry = _entry(inventory, "relay.git")
    assert entry.kind == "git-bare"
    assert entry.primary
    assert entry.source_state == "not-applicable"
    assert entry.retirement_blockers == (
        "ownership-unknown",
        "primary-worktree",
    )


def test_schema_one_registry_cannot_forge_managed_custody(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspaces": [
                    {
                        "path": "/tmp/forged",
                        "kind": "git-worktree",
                        "owner": "team:luce",
                        "custody": "managed",
                        "git_common_dir": "/tmp/repo/.git",
                        "git_dir": "/tmp/repo/.git/worktrees/forged",
                        "observed_head": "0" * 40,
                        "attested_at": "2026-07-15T00:00:00+00:00",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="does not grant managed custody"):
        WorkspaceRegistry(state_dir).load()


def test_every_ineligible_entry_reports_a_retirement_blocker(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _make_repository(work_root / "repo")

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 0)

    assert all(entry.retirement_eligible or entry.retirement_blockers for entry in inventory.entries)


@pytest.mark.parametrize(
    "operation_marker",
    ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_START", "rebase-merge", "rebase-apply", "sequencer"],
)
def test_inventory_marks_active_git_operations_busy(tmp_path, operation_marker):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    git_dir = Path(_git(primary, "rev-parse", "--absolute-git-dir"))
    marker = git_dir / operation_marker
    if "." in operation_marker or "_" in operation_marker:
        marker.write_text("active")
    else:
        marker.mkdir()

    inventory = inventory_workspaces(work_root, size_provider=lambda _path: 0)

    entry = _entry(inventory, "primary")
    assert entry.busy
    assert "git-operation-active" in entry.retirement_blockers


def test_inventory_reports_external_size_separately(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    primary = _make_repository(work_root / "primary")
    external = tmp_path / "external"
    _git(primary, "worktree", "add", "--detach", str(external), "HEAD")

    inventory = inventory_workspaces(work_root, size_provider=lambda path: 10 if path == external else 20)
    payload = inventory.to_dict()

    assert _entry(inventory, "external").size_bytes == 10
    assert inventory.total_size_bytes == 20
    assert payload["summary"]["external_size_bytes"] == 10
    assert payload["summary"]["external_size_unknown"] == 0


def test_inventory_does_not_measure_external_ancestor_of_work_root(tmp_path):
    repository = _make_repository(tmp_path / "repository")
    work_root = repository / "work"
    work_root.mkdir()
    linked = work_root / "linked"
    _git(repository, "worktree", "add", "--detach", str(linked), "HEAD")
    measured = []

    inventory = inventory_workspaces(work_root, size_provider=lambda path: measured.append(path) or 5)

    assert repository.resolve() not in measured
    assert _entry(inventory, "repository").size_known is False
    assert inventory.to_dict()["summary"]["external_size_unknown"] == 1


def test_dangling_symlink_is_measured_without_following_target(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    link = work_root / "missing-model"
    os.symlink(tmp_path / "does-not-exist", link)

    inventory = inventory_workspaces(work_root, size_provider=lambda path: path.lstat().st_size)

    entry = _entry(inventory, "missing-model")
    assert entry.kind == "symlink"
    assert entry.size_known
