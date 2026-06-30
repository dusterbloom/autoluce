"""
Living ideas queue: parse ROADMAP.md idea items and report which are untried.

Convention (documented in program.md): when an experiment targets a ROADMAP idea,
tag its description with the idea number, e.g. "[#3] adaptive K controller
(EAGLE-2, Li 2024)". This lets the agent see what's left without re-reading the
whole roadmap, and records the literature source inline for later PRs.

Advisory only — fuzzy by design; the agent still reads results.tsv for detail.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from profiling import ROADMAP_FOR_BOUND
from selector import rank_by_bottleneck

ROOT = Path(__file__).resolve().parent
ROADMAP = ROOT / "ROADMAP.md"
RESULTS_TSV = ROOT / "results.tsv"

_ITEM_RE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*")
_TAG_RE = re.compile(r"\[#(\d+)\]")


def load_roadmap_items(text: str) -> list[tuple[int, str]]:
    """Parse 'N. **Title.** ...' lines into (number, title) pairs."""
    items: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = _ITEM_RE.match(line)
        if m:
            items.append((int(m.group(1)), m.group(2).strip().rstrip(".")))
    return items


def extract_tried_numbers(descriptions: list[str]) -> set[int]:
    """Read '[#N]' tags from experiment descriptions."""
    tried: set[int] = set()
    for desc in descriptions:
        for m in _TAG_RE.finditer(desc):
            tried.add(int(m.group(1)))
    return tried


def untried(items: list[tuple[int, str]], tried_numbers: set[int]) -> list[tuple[int, str]]:
    """Return roadmap items whose number is not in tried_numbers."""
    return [(n, title) for n, title in items if n not in tried_numbers]


def _descriptions_from_results(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [row[-1] for row in csv.reader(f, delimiter="\t") if row]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report untried ROADMAP items, optionally ranked by the active bottleneck.",
    )
    parser.add_argument(
        "--bound", choices=list(ROADMAP_FOR_BOUND),
        help="Rank untried items so those targeting this bottleneck come first (from the profile verdict).",
    )
    args = parser.parse_args()

    items = load_roadmap_items(ROADMAP.read_text())
    tried = extract_tried_numbers(_descriptions_from_results(RESULTS_TSV))
    print(f"Tried: {sorted(tried) if tried else '(none)'}")
    remaining = untried(items, tried)
    if not remaining:
        print("All roadmap items tried. Re-profile and search literature for new ideas.")
        return 0

    ranked = rank_by_bottleneck(remaining, args.bound)
    header = f"Untried ({len(ranked)})"
    if args.bound:
        header += f" -- ranked for {args.bound}-bound workload"
    print(header + ":")
    for n, title, matched in ranked:
        marker = "  [matches bottleneck]" if (args.bound and matched) else ""
        print(f"  #{n}. {title}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
