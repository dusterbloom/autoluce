"""
Thin LLM proposer: ask an OpenAI-compatible model for the next experiment idea.

No-op until OPENAI_BASE_URL is set (see llm.config_from_env). When enabled, it ranks the
untried ROADMAP ideas by the profiling bottleneck (Move 1) and asks the model for one
concrete next experiment given the current best. This is ideation only -- the agent (or a
human) still writes experiment.py from the proposal; measurement / keep-revert stays in
agent_loop.py.
"""

from __future__ import annotations

import argparse
import sys

from autoluce.parallel.concurrency import LockedFrontier
from autoluce.ideation.ideas import descriptions_from_results, extract_tried_numbers, load_roadmap_items, untried
from autoluce.ideation.llm import complete, config_from_env
from autoluce.bench.profiling import ROADMAP_FOR_BOUND
from autoluce.ideation.selector import rank_by_bottleneck

from autoluce import ROOT
ROADMAP = ROOT / "ROADMAP.md"
RESULTS_TSV = ROOT / "results.tsv"


def build_proposal_messages(untried_items: list[tuple[int, str]], best_score: float, bound: str | None = None) -> list[dict]:
    """Pure: render the harness state into a chat prompt asking for the next idea."""
    if untried_items:
        idea_lines = "\n".join(f"  #{n}. {t}" for n, t in untried_items)
    else:
        idea_lines = "  (none -- re-profile and mine literature for new ideas)"
    bound_note = f"\nThe profile says decode is {bound}-bound; prioritize ideas targeting that wall." if bound else ""
    user = (
        f"Current best score: {best_score}\n"
        f"Untried ROADMAP ideas:\n{idea_lines}{bound_note}\n\n"
        f"Propose the single next experiment to try. Give a one-line rationale and the "
        "concrete change (product CMake flag, runtime flag, or Lucebox Hub/vendor patch target)."
    )
    return [
        {"role": "system", "content": (
            "You are a GPU inference optimization researcher working on a speculative-decoding "
            "ggml engine. Propose ONE minimal, concrete experiment at a time."
        )},
        {"role": "user", "content": user},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask an LLM for the next experiment idea.")
    parser.add_argument("--bound", choices=list(ROADMAP_FOR_BOUND),
                        help="Profile verdict; ranks ideas and tells the model the wall.")
    args = parser.parse_args()

    config = config_from_env()
    if config is None:
        print("LLM disabled. Set OPENAI_BASE_URL to enable (OPENAI_API_KEY for cloud, AUTOLUCE_MODEL to override).")
        return 0

    items = load_roadmap_items(ROADMAP.read_text())
    tried = extract_tried_numbers(descriptions_from_results(RESULTS_TSV))
    ranked = rank_by_bottleneck(untried(items, tried), args.bound)
    pairs = [(n, t) for n, t, _ in ranked]
    best = LockedFrontier(ROOT).read_best().get("score", 0.0)

    print(complete(build_proposal_messages(pairs, best, args.bound), config=config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
