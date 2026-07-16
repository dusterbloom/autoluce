"""Quality-constrained Pareto archive over immutable campaign evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from autoluce.research_evidence import CampaignEvidence


class ParetoArchive:
    def __init__(self, objectives: Mapping[str, str]) -> None:
        if not objectives:
            raise ValueError("frontier requires at least one objective")
        invalid = sorted(set(objectives.values()) - {"maximize", "minimize"})
        if invalid:
            raise ValueError("frontier directions must be maximize or minimize")
        self.objectives = dict(objectives)
        self._evidence: list[CampaignEvidence] = []

    @property
    def evidence(self) -> tuple[CampaignEvidence, ...]:
        return tuple(self._evidence)

    def add(self, evidence: CampaignEvidence) -> None:
        if any(item.evidence_id == evidence.evidence_id for item in self._evidence):
            return
        self._evidence.append(evidence)

    def _dominates(self, left: CampaignEvidence, right: CampaignEvidence) -> bool:
        at_least_as_good = True
        strictly_better = False
        for metric, direction in self.objectives.items():
            if metric not in left.metrics or metric not in right.metrics:
                return False
            a, b = float(left.metrics[metric]), float(right.metrics[metric])
            better_or_equal = a >= b if direction == "maximize" else a <= b
            better = a > b if direction == "maximize" else a < b
            at_least_as_good = at_least_as_good and better_or_equal
            strictly_better = strictly_better or better
        return at_least_as_good and strictly_better

    @property
    def frontier(self) -> tuple[CampaignEvidence, ...]:
        # Compatibility cells cannot dominate one another.  This preserves a
        # machine/workload frontier without ever comparing unlike evidence.
        cells: dict[str, list[CampaignEvidence]] = defaultdict(list)
        for item in self._evidence:
            if item.feasible and all(metric in item.metrics for metric in self.objectives):
                cells[item.compatibility_key].append(item)
        frontier = []
        for items in cells.values():
            frontier.extend(
                item for item in items
                if not any(other is not item and self._dominates(other, item) for other in items)
            )
        return tuple(frontier)
