"""
Idea selection: rank untried ROADMAP items by relevance to the active bottleneck.

This is the testable core of profile-driven ideation (ROADMAP Meta / Move 1). The
profiler's trace parser (which produces the bound verdict + per-phase time fractions
for true Amdahl weighting) is a separate I/O concern; this function consumes the bound
as given, so the ranking decision stays pure and deterministic.

Relevance (bound -> ROADMAP item numbers) defaults to profiling._ROADMAP_FOR_BOUND so
the mapping lives in exactly one place.
"""

from __future__ import annotations

from profiling import _ROADMAP_FOR_BOUND


def rank_by_bottleneck(
    items: list[tuple[int, str]],
    bound: str | None,
    relevance: dict[str, list[int]] | None = None,
) -> list[tuple[int, str, bool]]:
    """Reorder `items` so those targeting the active bottleneck come first.

    Matched items (whose number is in relevance[bound]) precede off-bottleneck items;
    each group keeps numerical order. bound=None or unknown -> numerical order unchanged,
    all flagged unmatched. Returns (number, title, matched) triples so the caller can
    annotate the worklist.
    """
    if relevance is None:
        relevance = _ROADMAP_FOR_BOUND
    matching = set(relevance[bound]) if bound in relevance else set()
    matched = [(n, t, True) for n, t in items if n in matching]
    off = [(n, t, False) for n, t in items if n not in matching]
    return matched + off
