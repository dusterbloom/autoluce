from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autoluce.pr_overlap import (
    GitHubPullRequestClient,
    PullRequest,
    changed_paths,
    find_overlaps,
    focus_terms,
    pull_request_matches_focus,
    repository_slug,
)


def _pull_request(number: int, title: str, files: tuple[str, ...]) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        url=f"https://github.com/Luce-Org/lucebox/pull/{number}",
        base="main",
        head=f"feature-{number}",
        draft=False,
        body="",
        files=files,
    )


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def test_repository_slug_accepts_https_and_ssh_urls():
    assert repository_slug("https://github.com/Luce-Org/lucebox-hub.git") == "Luce-Org/lucebox-hub"
    assert repository_slug("git@github.com:Luce-Org/lucebox.git") == "Luce-Org/lucebox"
    with pytest.raises(ValueError, match="unsupported GitHub repository"):
        repository_slug("https://gitlab.com/Luce-Org/lucebox")


def test_ds4_focus_matches_deepseek4_titles_and_paths():
    pull_request = _pull_request(
        514,
        "Restore DeepSeek4 HIP backend safely",
        ("server/src/deepseek4/deepseek4_graph.cpp",),
    )
    assert focus_terms("ds4") == ("deepseek4", "ds4")
    assert pull_request_matches_focus(pull_request, "ds4")
    assert pull_request_matches_focus(pull_request, "hip")
    assert not pull_request_matches_focus(pull_request, "qwen")


def test_find_overlaps_reports_exact_paths_in_pr_order():
    graph = "server/src/deepseek4/deepseek4_graph.cpp"
    pool = "server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
    overlaps = find_overlaps(
        (graph, pool, "docs/note.md"),
        (
            _pull_request(502, "ROCmFPX", (pool,)),
            _pull_request(514, "DS4 prefill", (graph,)),
            _pull_request(513, "MMQ", ("server/deps/llama.cpp/ggml/src/ggml-cuda/mmq.cu",)),
        ),
    )
    assert [(item.pull_request.number, item.paths) for item in overlaps] == [
        (514, (graph,)),
        (502, (pool,)),
    ]


def test_changed_paths_combines_branch_dirty_and_untracked_files(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@autoluce.local")
    _git(repository, "config", "user.name", "AutoLuce Test")
    (repository / "base.cpp").write_text("base\n")
    (repository / "dirty.cpp").write_text("clean\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (repository / "base.cpp").write_text("branch\n")
    _git(repository, "add", "base.cpp")
    _git(repository, "commit", "-qm", "candidate")
    (repository / "dirty.cpp").write_text("dirty\n")
    (repository / "untracked.cpp").write_text("new\n")

    assert changed_paths(repository, base) == ["base.cpp", "dirty.cpp", "untracked.cpp"]
    assert changed_paths(repository, base, include_dirty=False) == ["base.cpp"]


def test_changed_paths_explains_unreachable_base(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@autoluce.local")
    _git(repository, "config", "user.name", "AutoLuce Test")
    (repository / "file.cpp").write_text("base\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "base")

    with pytest.raises(RuntimeError, match="fetch the required history"):
        changed_paths(repository, "f" * 40)


def test_github_client_prefers_single_gh_inventory_call():
    payload = [
        {
            "number": 514,
            "title": "Restore DeepSeek4 HIP backend safely",
            "url": "https://github.com/Luce-Org/lucebox/pull/514",
            "baseRefName": "main",
            "headRefName": "ds4_prefill_batch",
            "isDraft": True,
            "body": "batched prefill",
            "files": [{"path": "server/src/deepseek4/deepseek4_graph.cpp"}],
        }
    ]
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    pull_requests = GitHubPullRequestClient(runner=runner).list_open(
        "https://github.com/Luce-Org/lucebox-hub.git"
    )

    assert len(calls) == 1
    assert calls[0][:4] == ["gh", "pr", "list", "--repo"]
    assert pull_requests == [
        PullRequest(
            number=514,
            title="Restore DeepSeek4 HIP backend safely",
            url="https://github.com/Luce-Org/lucebox/pull/514",
            base="main",
            head="ds4_prefill_batch",
            draft=True,
            body="batched prefill",
            files=("server/src/deepseek4/deepseek4_graph.cpp",),
        )
    ]
