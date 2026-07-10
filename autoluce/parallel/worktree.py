"""
Git worktree lifecycle: an isolated source tree per worker, sharing one object store.

Each parallel worker gets its own working tree + build dir, so N concurrent builds no
longer clobber one shared Lucebox product build. This is the local-host equivalent of
runner.run_fn's remote/VM isolation: a worker built on a worktree is as isolated as one
built on a separate box, but without the second clone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

WORKTREE_DIR_NAME = ".worktrees"


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def ensure_worktree(repo_root: Path, name: str, commit: str = "HEAD") -> Path:
    """Create a worktree of `repo_root` at `.worktrees/<name>` checked out to `commit`.

    Idempotent: if the worktree already exists it is returned as-is. Callers needing a
    different commit should remove_worktree first. Returns the worktree path.
    """
    wt_root = repo_root / WORKTREE_DIR_NAME
    wt_root.mkdir(parents=True, exist_ok=True)
    path = wt_root / name
    if path.exists():
        return path
    _run(["git", "worktree", "add", str(path), commit], cwd=repo_root)
    return path


def remove_worktree(repo_root: Path, name: str) -> None:
    """Remove a worktree created by ensure_worktree. No-op if it does not exist."""
    path = repo_root / WORKTREE_DIR_NAME / name
    if not path.exists():
        return
    _run(["git", "worktree", "remove", "--force", str(path)], cwd=repo_root)
