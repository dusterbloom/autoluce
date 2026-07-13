"""Discover open pull requests that overlap a candidate Lucebox change."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

import requests


FOCUS_ALIASES: dict[str, tuple[str, ...]] = {
    "ds4": ("ds4", "deepseek4"),
    "deepseek4": ("ds4", "deepseek4"),
    "hip": ("hip", "rocm", "gfx1151"),
    "rocm": ("hip", "rocm", "gfx1151"),
}


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    url: str
    base: str
    head: str
    draft: bool
    body: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class PullRequestOverlap:
    pull_request: PullRequest
    paths: tuple[str, ...]


def repository_slug(repository: str) -> str:
    """Return ``owner/name`` for an HTTPS or SSH GitHub repository URL."""
    value = repository.strip().removesuffix(".git").rstrip("/")
    if value.startswith("git@github.com:"):
        slug = value.split(":", 1)[1]
    else:
        parsed = urlparse(value)
        if parsed.hostname != "github.com":
            raise ValueError(f"unsupported GitHub repository: {repository}")
        slug = parsed.path.strip("/")
    if len(slug.split("/")) != 2:
        raise ValueError(f"expected a GitHub owner/repository pair: {repository}")
    return slug


def focus_terms(focus: str | None) -> tuple[str, ...]:
    if not focus:
        return ()
    terms: set[str] = set()
    for raw in focus.split(","):
        term = raw.strip().lower()
        if not term:
            continue
        terms.update(FOCUS_ALIASES.get(term, (term,)))
    return tuple(sorted(terms))


def pull_request_matches_focus(pull_request: PullRequest, focus: str | None) -> bool:
    terms = focus_terms(focus)
    if not terms:
        return True
    searchable = "\n".join(
        (
            pull_request.title,
            pull_request.body,
            pull_request.base,
            pull_request.head,
            *pull_request.files,
        )
    ).lower()
    return any(term in searchable for term in terms)


def find_overlaps(
    candidate_paths: Iterable[str],
    pull_requests: Iterable[PullRequest],
) -> list[PullRequestOverlap]:
    candidate = set(candidate_paths)
    overlaps = []
    for pull_request in pull_requests:
        shared = tuple(sorted(candidate.intersection(pull_request.files)))
        if shared:
            overlaps.append(PullRequestOverlap(pull_request, shared))
    return sorted(overlaps, key=lambda item: item.pull_request.number, reverse=True)


def _git_paths(
    repository: Path,
    args: list[str],
    runner: Callable = subprocess.run,
) -> set[str]:
    process = runner(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        stderr = process.stderr.decode(errors="replace").strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return {value.decode(errors="replace") for value in process.stdout.split(b"\0") if value}


def changed_paths(
    repository: Path,
    base: str,
    *,
    include_dirty: bool = True,
    runner: Callable = subprocess.run,
) -> list[str]:
    """Return committed branch changes plus optional staged, dirty, and untracked paths."""
    merge_base = runner(
        ["git", "merge-base", base, "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode:
        detail = merge_base.stderr.strip()
        message = f"cannot find a merge base between {base} and HEAD"
        if detail:
            message += f": {detail}"
        message += "; fetch the required history or pass a reachable --base"
        raise RuntimeError(message)
    ancestor = merge_base.stdout.strip()
    paths = _git_paths(repository, ["diff", "--name-only", "-z", ancestor, "HEAD"], runner)
    if include_dirty:
        paths.update(_git_paths(repository, ["diff", "--name-only", "-z", "HEAD"], runner))
        paths.update(
            _git_paths(repository, ["ls-files", "--others", "--exclude-standard", "-z"], runner)
        )
    return sorted(paths)


class GitHubPullRequestClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        token: str | None = None,
        runner: Callable = subprocess.run,
    ):
        self.session = session or requests.Session()
        self.runner = runner
        self.token = token or next(
            (
                os.environ[name]
                for name in ("AUTOLUCE_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
                if os.environ.get(name)
            ),
            None,
        )

    @staticmethod
    def _from_gh_payload(raw: dict) -> PullRequest:
        return PullRequest(
            number=int(raw["number"]),
            title=str(raw["title"]),
            url=str(raw["url"]),
            base=str(raw["baseRefName"]),
            head=str(raw["headRefName"]),
            draft=bool(raw.get("isDraft", False)),
            body=str(raw.get("body") or ""),
            files=tuple(sorted(str(item["path"]) for item in raw.get("files", []))),
        )

    def _list_open_with_gh(self, repository: str) -> list[PullRequest] | None:
        fields = "number,title,url,baseRefName,headRefName,isDraft,body,files"
        try:
            process = self.runner(
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    repository_slug(repository),
                    "--state",
                    "open",
                    "--limit",
                    "100",
                    "--json",
                    fields,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if process.returncode:
            return None
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        return sorted((self._from_gh_payload(raw) for raw in payload), key=lambda item: item.number, reverse=True)

    def _get_pages(self, url: str) -> Iterable[dict]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        page = 1
        while True:
            response = self.session.get(
                url,
                headers=headers,
                params={"per_page": 100, "page": page},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"unexpected GitHub response from {url}")
            yield from payload
            if len(payload) < 100:
                return
            page += 1

    def list_open(self, repository: str) -> list[PullRequest]:
        from_gh = self._list_open_with_gh(repository)
        if from_gh is not None:
            return from_gh
        slug = repository_slug(repository)
        api_root = f"https://api.github.com/repos/{slug}"
        pull_requests = []
        for raw in self._get_pages(f"{api_root}/pulls?state=open"):
            number = int(raw["number"])
            files = tuple(
                sorted(
                    str(item["filename"])
                    for item in self._get_pages(f"{api_root}/pulls/{number}/files")
                )
            )
            pull_requests.append(
                PullRequest(
                    number=number,
                    title=str(raw["title"]),
                    url=str(raw["html_url"]),
                    base=str(raw["base"]["ref"]),
                    head=str(raw["head"]["ref"]),
                    draft=bool(raw.get("draft", False)),
                    body=str(raw.get("body") or ""),
                    files=files,
                )
            )
        return sorted(pull_requests, key=lambda item: item.number, reverse=True)
