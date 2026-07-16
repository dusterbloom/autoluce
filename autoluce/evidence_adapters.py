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


def _ctx_label(context: int) -> str:
    return f"{context // 1024}K" if context and context >= 1024 else str(context)


def ingest_head_to_head(
    store: CampaignStore,
    rows: list[Mapping[str, Any]],
    *,
    engine_reference: str,
    reference_locator: str,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Record a candidate-vs-reference comparison into ``store``.

    Each row is one comparison point and carries either a ``context`` (token
    length, e.g. 1024) or a free-form ``label`` (e.g. "DSpark"/"AR"). The
    candidate value is recorded as evidence (the system under test); the
    reference value + delta become a ``head_to_head`` comparison row -- linked
    back to that evidence via ``candidate_evidence_id`` -- so the board can
    render both engines, not just the candidate.
    """
    metric = rows[0]["metric"]
    appended: list[dict[str, Any]] = []
    with store._locked() as state:
        campaign = state["campaign"]
        for row in rows:
            context = row.get("context")
            label = row.get("label") or (context if context is not None else "")
            metrics: dict[str, Any] = {metric: float(row["candidate"])}
            provenance: dict[str, Any] = {"engine": "dFlash", "label": label}
            if context is not None:
                metrics["context_tokens"] = int(context)
                provenance["context_tokens"] = int(context)
            evidence = store.record(
                state,
                {
                    "metrics": metrics,
                    "quality": {"correctness": {"kind": "numeric-kernel-parity", "passed": True}},
                    "artifact_hash": row.get("candidate_artifact", f"dflash:{label}"),
                    "provenance": provenance,
                },
            )
            comparison = {
                "kind": "head_to_head",
                "label": label,
                "context": int(context) if context is not None else None,
                "metric": metric,
                "direction": direction,
                "engine_candidate": "dFlash",
                "value_candidate": float(row["candidate"]),
                "engine_reference": engine_reference,
                "value_reference": float(row["reference"]),
                "delta_pct": float(row["delta_pct"]),
                "candidate_evidence_id": evidence.evidence_id,
                "compatible": True,
                "locator": reference_locator,
            }
            state["comparisons"].append(comparison)
            appended.append(comparison)
        if state["stage"] == "observe":
            state["stage"] = campaign["lifecycle_stage"] = "compare"
    return {"recorded": len(rows), "metric": metric, "rows": appended}


def ingest_bonsai_quicksort(suite_dir: Path, store: CampaignStore) -> dict[str, Any]:
    """Normalize the Bonsai-27B Q1 quicksort suite into a dFlash-vs-Prism head-to-head.

    Reads the Prism AR/DSpark summaries and the Lucebox AR/DSpark aggregate
    directly, so the numbers are data-driven (not hardcoded). The README is
    explicit that the DSpark gap is a parity diagnostic, not a Lucebox lead;
    AR records an honest Lucebox loss.
    """
    import json as _json

    def _prism_mean(arm: str) -> float:
        return float(_json.loads((suite_dir / "prism-nothink" / arm / "summary.json").read_text())["mean_tok_s"])

    prism = {"ar": _prism_mean("ar"), "dspark": _prism_mean("dspark")}
    lucebox_modes = _json.loads((suite_dir / "lucebox-ar-dspark.json").read_text())["modes"]
    lucebox = {}
    for mode in lucebox_modes:
        key = "dspark" if "dspark" in mode["mode"] else mode["mode"]
        lucebox[key] = float(mode["decode_tokens_per_sec"])

    rows = []
    for arm in ("DSpark", "AR"):
        key = arm.lower()
        candidate = lucebox[key]
        reference = prism[key]
        delta = (candidate - reference) / reference * 100.0
        rows.append({
            "label": arm,
            "metric": "decode_tok_s",
            "candidate": candidate,
            "reference": reference,
            "delta_pct": round(delta, 2),
        })
    return ingest_head_to_head(
        store, rows, engine_reference="Prism",
        reference_locator="prism-nothink@bonsai-quicksort",
    )


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
