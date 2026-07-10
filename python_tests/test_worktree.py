"""
Tests for git worktree lifecycle helpers.

Worktrees give each parallel worker its own source tree + build dir while sharing
one object store -- the local isolation that makes N concurrent builds safe on a
single host (the remote/VM case is already covered by runner's injected run_fn).
"""

import subprocess
from pathlib import Path

from autoluce.parallel.worktree import ensure_worktree, remove_worktree


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "t@t.t"], repo)
    _run(["git", "config", "user.name", "t"], repo)
    (repo / "marker.txt").write_text("hi")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-qm", "init"], repo)
    return repo


def test_ensure_worktree_creates_real_checkout(tmp_path):
    repo = _make_repo(tmp_path)
    wt = ensure_worktree(repo, "w1")
    assert wt.exists()
    # A worktree is a real checkout of the commit: the committed file is present.
    assert (wt / "marker.txt").read_text() == "hi"
    assert (wt / ".git").exists()  # worktree's gitdir pointer


def test_ensure_worktree_is_idempotent(tmp_path):
    repo = _make_repo(tmp_path)
    first = ensure_worktree(repo, "w1")
    second = ensure_worktree(repo, "w1")
    assert first == second


def test_remove_worktree_cleans_up(tmp_path):
    repo = _make_repo(tmp_path)
    wt = ensure_worktree(repo, "w1")
    assert wt.exists()
    remove_worktree(repo, "w1")
    assert not wt.exists()


def test_remove_worktree_is_noop_when_absent(tmp_path):
    repo = _make_repo(tmp_path)
    remove_worktree(repo, "never-existed")  # must not raise
