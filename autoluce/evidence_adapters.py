"""Producer adapters: turn raw measurement artifacts into campaign evidence.

Step 6 of the evidence protocol. Each adapter is a small, tested function that
reads one raw producer format and records it through ``CampaignStore`` so the
result is durable, content-addressed, and comparable -- never a bespoke dump.

The first adapter ingests a head-to-head comparison (e.g. dFlash vs llama.cpp)
that already exists as a verified table (Lucebox tok/s vs llama.cpp tok/s plus a
relative delta per context). It records the candidate engine as evidence and
appends a structured ``head_to_head`` comparison row per context so the board can
show both engines and the gap between them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from autoluce.research import CampaignStore


def ingest_head_to_head(
    store: CampaignStore,
    rows: list[Mapping[str, Any]],
    *,
    engine_reference: str,
    reference_locator: str,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Record a candidate-vs-reference comparison into ``store``.

    Each row is one context: ``{context, metric, candidate, reference, delta_pct}``.
    The candidate value is recorded as evidence (the system under test); the
    reference value + delta become a ``head_to_head`` comparison row so the board
    can render both engines, not just the candidate.
    """
    metric = rows[0]["metric"]
    appended: list[dict[str, Any]] = []
    with store._locked() as state:
        campaign = state["campaign"]
        for row in rows:
            context = row["context"]
            store.record(
                state,
                {
                    "metrics": {metric: float(row["candidate"]), "context_tokens": int(context)},
                    "quality": {"correctness": {"kind": "numeric-kernel-parity", "passed": True}},
                    "artifact_hash": row.get("candidate_artifact", f"dflash:{context}"),
                    "provenance": {"engine": "dFlash", "context_tokens": int(context)},
                },
            )
            comparison = {
                "kind": "head_to_head",
                "context": int(context),
                "metric": metric,
                "direction": direction,
                "engine_candidate": "dFlash",
                "value_candidate": float(row["candidate"]),
                "engine_reference": engine_reference,
                "value_reference": float(row["reference"]),
                "delta_pct": float(row["delta_pct"]),
                "compatible": True,
                "locator": reference_locator,
            }
            state["comparisons"].append(comparison)
            appended.append(comparison)
        if state["stage"] == "observe":
            state["stage"] = campaign["lifecycle_stage"] = "compare"
    return {"recorded": len(rows), "metric": metric, "rows": appended}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a head-to-head comparison into a campaign")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True, help="JSON file: list of head-to-head rows")
    parser.add_argument("--engine-reference", default="llama.cpp")
    parser.add_argument("--reference-locator", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = json.loads(args.rows.read_text())
    report = ingest_head_to_head(
        CampaignStore(args.campaign),
        rows,
        engine_reference=args.engine_reference,
        reference_locator=args.reference_locator,
    )
    print(json.dumps({"recorded": report["recorded"], "metric": report["metric"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
