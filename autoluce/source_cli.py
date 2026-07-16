"""Inspect the Lucebox product pin, vendored provenance, and upstream drift."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from autoluce.pr_overlap import (
    GitHubPullRequestClient,
    changed_paths,
    find_overlaps,
    pull_request_matches_focus,
)
from autoluce.source_layout import SourceLayout, SourceManifest, check_remote_drift


def _pull_request_payload(pull_request) -> dict:
    return {
        "number": pull_request.number,
        "title": pull_request.title,
        "url": pull_request.url,
        "base": pull_request.base,
        "head": pull_request.head,
        "draft": pull_request.draft,
        "files": list(pull_request.files),
    }


def _print_pull_request(pull_request) -> None:
    state = "draft" if pull_request.draft else "open"
    print(f"  #{pull_request.number} [{state}] {pull_request.title}")
    print(f"    {pull_request.base} <- {pull_request.head}; {len(pull_request.files)} files")
    print(f"    {pull_request.url}")


def main(argv: list[str] | None = None, client=None) -> None:
    parser = argparse.ArgumentParser(prog="autoluce source")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="show configured and checked-out source ownership")
    status.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="fail when the tracked Lucebox branch moved")
    check.add_argument("--remote", action="store_true", required=True)
    check.add_argument("--json", action="store_true")
    prs = commands.add_parser("prs", help="list open pull requests before starting overlapping work")
    prs.add_argument("--focus", help="limit results by topic; ds4 also matches deepseek4")
    prs.add_argument("--json", action="store_true")
    overlap = commands.add_parser("overlap", help="compare candidate paths with open pull requests")
    overlap.add_argument("--base", help="candidate base revision; defaults to the pinned product ref")
    overlap.add_argument("--focus", help="also report related pull requests by topic")
    overlap.add_argument("--no-dirty", action="store_true", help="ignore uncommitted and untracked paths")
    overlap.add_argument("--fail-on-overlap", action="store_true")
    overlap.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    manifest = SourceManifest.load()
    if args.command == "check":
        drift = check_remote_drift(manifest)
        payload = asdict(drift)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            state = "update available" if drift.changed else "current"
            print(f"Lucebox source: {state}\n  pinned:   {drift.pinned}\n  upstream: {drift.upstream}")
        raise SystemExit(3 if drift.changed else 0)

    if args.command in ("prs", "overlap"):
        github = client or GitHubPullRequestClient()
        pull_requests = github.list_open(manifest.repository)
        focused = [pr for pr in pull_requests if pull_request_matches_focus(pr, args.focus)]
        if args.command == "prs":
            payload = {
                "repository": manifest.repository,
                "focus": args.focus,
                "pull_requests": [_pull_request_payload(pr) for pr in focused],
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                label = f" matching {args.focus!r}" if args.focus else ""
                print(f"Open Lucebox pull requests{label}: {len(focused)}")
                for pull_request in focused:
                    _print_pull_request(pull_request)
            return

        layout = SourceLayout.resolve()
        base = args.base or manifest.ref
        candidate_paths = changed_paths(layout.checkout, base, include_dirty=not args.no_dirty)
        exact = find_overlaps(candidate_paths, pull_requests)
        exact_numbers = {item.pull_request.number for item in exact}
        related = [pr for pr in focused if pr.number not in exact_numbers]
        payload = {
            "repository": manifest.repository,
            "checkout": str(layout.checkout),
            "base": base,
            "include_dirty": not args.no_dirty,
            "candidate_paths": candidate_paths,
            "overlaps": [
                {"pull_request": _pull_request_payload(item.pull_request), "paths": list(item.paths)}
                for item in exact
            ],
            "related": [_pull_request_payload(pr) for pr in related],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Candidate paths: {len(candidate_paths)} (base {base})")
            print(f"Exact open-PR overlaps: {len(exact)}")
            for item in exact:
                _print_pull_request(item.pull_request)
                for path in item.paths:
                    print(f"      {path}")
            if args.focus:
                print(f"Related open PRs matching {args.focus!r}: {len(related)}")
                for pull_request in related:
                    _print_pull_request(pull_request)
        if args.fail_on_overlap and exact:
            raise SystemExit(4)
        return

    layout = SourceLayout.resolve()
    checked_out = layout.checkout.exists()
    provenance = None
    if checked_out:
        provenance = asdict(layout.validate())
    payload = {
        "repository": manifest.repository,
        "ref": manifest.ref,
        "track": manifest.track,
        "layout": manifest.layout,
        "checkout": str(layout.checkout),
        "checked_out": checked_out,
        "runtime": manifest.runtime,
        "capabilities": manifest.capabilities,
        "product_backends": manifest.product_backends,
        "vendor_backends": manifest.vendor_backends,
        "submodules_by_backend": manifest.submodules_by_backend,
        "vendor": provenance or asdict(manifest.vendor),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Lucebox source\n  repository: {manifest.repository}\n  pin:        {manifest.ref}")
        print(f"  layout:     {manifest.layout}\n  checkout:   {layout.checkout} ({'ready' if checked_out else 'not cloned'})")
        print(f"  runtime:    {manifest.runtime}\n  vendor:     {manifest.vendor.base_commit}")
        print(f"  product:    {', '.join(manifest.product_backends)}\n  vendor API: {', '.join(manifest.vendor_backends)}")


if __name__ == "__main__":
    main()
