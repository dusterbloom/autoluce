"""Zero-dependency read-only WebUI over AutoLuce campaign state.

This module is a thin viewer. Campaign layout, evidence, frontier, and stage
are owned by ``CampaignStore``; the WebUI never re-derives those conventions
or invents its own status model. It discovers candidate campaign contracts,
hands each to ``CampaignStore.load()``, and reports what the authoritative
state says.

When mounted inside the coordinator HTTP server the WebUI owns the browser
surface (``/``, ``/static/*``, ``/api/campaigns/*``); the coordinator owns
``/v1/*``. Routing is decided by explicit path predicates, not recovered
from dispatch errors.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autoluce import ROOT
from autoluce.research import CampaignStore


def _static_dir() -> Path:
    return Path(__file__).with_suffix("").parent / "static"


def _resolve_root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else ROOT


def _discover_campaign_paths(root: Path | str | None = None) -> list[Path]:
    base = _resolve_root(root)
    paths: list[Path] = []
    research = base / ".autoluce" / "research"
    if research.exists():
        paths.extend(sorted(research.rglob("campaign.json")))
    benchmarks = base / "benchmarks"
    if benchmarks.exists():
        paths.extend(sorted(benchmarks.rglob("campaigns/*.json")))
    return paths


def _load_state(path: Path) -> dict[str, Any] | None:
    try:
        return CampaignStore(path).load()
    except Exception:
        return None


def _status(state: dict[str, Any]) -> str:
    if state.get("promotion"):
        return "promoted"
    if state.get("evidence"):
        return "measured"
    return "planned"


def _lineage(path: Path, base: Path) -> tuple[str, str, str]:
    """Derive (root, family, variant) from a campaign's path.

    The WebUI does not invent a taxonomy; it reports the researcher's own
    directory tree. ``family`` is the top family (or benchmark suite);
    ``variant`` is the deeper sub-path label, empty when the family has a
    single campaign.
    """
    parts = path.relative_to(base).parts
    if len(parts) >= 3 and parts[0] == ".autoluce" and parts[1] == "research":
        return "research", parts[2], "/".join(parts[3:-1])
    if parts and parts[0] == "benchmarks":
        family = parts[1] if len(parts) > 1 else "benchmarks"
        return "benchmarks", family, path.stem
    family = parts[0] if parts else path.stem
    return "research", family, "/".join(parts[1:-1])


def _metric_series(campaign: dict[str, Any], evidence: list[dict[str, Any]]) -> list[float]:
    metric = campaign.get("objective", {}).get("metric")
    if not metric:
        return []
    series: list[float] = []
    for item in evidence:
        value = item.get("metrics", {}).get(metric)
        if value is not None:
            series.append(float(value))
    return series


def _campaign_view(path: Path, base: Path) -> dict[str, Any] | None:
    state = _load_state(path)
    if state is None:
        return None
    campaign = state["campaign"]
    campaign_id = state["campaign_id"]
    root, family, variant = _lineage(path, base)
    return {
        "id": campaign_id,
        "campaign_id": campaign_id,
        "name": campaign["name"],
        "stage": campaign["lifecycle_stage"],
        "status": _status(state),
        "path": str(path.relative_to(base)),
        "root": root,
        "family": family,
        "variant": variant,
        "system": campaign["system"],
        "objective": campaign["objective"],
        "evidence_count": len(state["evidence"]),
        "frontier_count": len(state["frontier"]),
        "reference_count": len(state.get("references", [])),
        "promotion": state.get("promotion"),
        "sparkline": _metric_series(campaign, state["evidence"]),
    }


def list_campaigns(root: Path | str | None = None) -> list[dict[str, Any]]:
    base = _resolve_root(root)
    views = [view for view in (_campaign_view(path, base) for path in _discover_campaign_paths(base)) if view]
    views.sort(key=lambda view: view["name"])
    return views


def _plot_data(campaign: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    metric = campaign.get("objective", {}).get("metric")
    if not metric:
        return None
    points: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        value = item.get("metrics", {}).get(metric)
        if value is None:
            continue
        points.append({"index": index, "value": float(value), "evidence_id": item.get("evidence_id", "")})
    if not points:
        return None
    values = [point["value"] for point in points]
    return {
        "metric": metric,
        "unit": "tokens/s" if str(metric).endswith("_tok_s") else "",
        "points": points,
        "min": min(values),
        "max": max(values),
    }


def get_campaign(campaign_id: str, root: Path | str | None = None) -> dict[str, Any] | None:
    base = _resolve_root(root)
    for path in _discover_campaign_paths(base):
        state = _load_state(path)
        if state is None or state.get("campaign_id") != campaign_id:
            continue
        campaign = state["campaign"]
        return {
            "campaign_id": campaign_id,
            "stage": campaign["lifecycle_stage"],
            "status": _status(state),
            "path": str(path.relative_to(base)),
            "campaign": campaign,
            "state": state,
            "plot": _plot_data(campaign, state["evidence"]),
        }
    return None


def is_webui_request(path: str) -> bool:
    parsed = urlparse(path).path
    return parsed == "/" or parsed.startswith("/static/") or parsed.startswith("/api/campaigns")


def is_public_webui(path: str) -> bool:
    parsed = urlparse(path).path
    return parsed == "/" or parsed.startswith("/static/")


def _json_response(handler: Any, status: int, payload: Any) -> None:
    body = json.dumps(payload, indent=2, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _file_response(handler: Any, path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ValueError(f"not found: {path}")
    data = path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(path))
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def web_dispatch(handler: Any) -> None:
    """Serve WebUI routes for a ``BaseHTTPRequestHandler`` instance."""
    if handler.command != "GET":
        raise ValueError("WebUI is read-only; only GET is supported")

    path = urlparse(handler.path).path

    if path == "/":
        return _file_response(handler, _static_dir() / "index.html")

    if path.startswith("/static/"):
        target = _static_dir() / path[len("/static/"):]
        try:
            target.resolve().relative_to(_static_dir().resolve())
        except ValueError:
            raise ValueError("invalid static path")
        return _file_response(handler, target)

    if path == "/api/campaigns":
        return _json_response(handler, 200, list_campaigns())

    if path.startswith("/api/campaigns/"):
        campaign_id = path[len("/api/campaigns/"):]
        detail = get_campaign(campaign_id)
        if detail is None:
            return _json_response(handler, 404, {"error": "campaign not found"})
        return _json_response(handler, 200, detail)

    raise ValueError(f"unsupported WebUI path: {path}")


def run_standalone(host: str = "127.0.0.1", port: int = 8766) -> None:
    """Run the read-only WebUI on its own for local development."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            try:
                web_dispatch(self)
            except ValueError as error:
                _json_response(self, 404, {"error": str(error)})
            except Exception as error:
                _json_response(self, 500, {"error": str(error)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"autoluce webui listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Browse AutoLuce campaigns in a browser (read-only, no auth)"
    )
    parser.add_argument("--listen", default="127.0.0.1", help="bind address (default localhost)")
    parser.add_argument("--port", type=int, default=8766, help="bind port")
    args = parser.parse_args(argv)
    run_standalone(host=args.listen, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
