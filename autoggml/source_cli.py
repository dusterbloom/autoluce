"""Inspect the Lucebox product pin, vendored provenance, and upstream drift."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from autoggml.source_layout import SourceLayout, SourceManifest, check_remote_drift


def main() -> None:
    parser = argparse.ArgumentParser(prog="autoggml source")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="show configured and checked-out source ownership")
    status.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="fail when the tracked Lucebox branch moved")
    check.add_argument("--remote", action="store_true", required=True)
    check.add_argument("--json", action="store_true")
    args = parser.parse_args()

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
        "supported_backends": manifest.supported_backends,
        "submodules_by_backend": manifest.submodules_by_backend,
        "vendor": provenance or asdict(manifest.vendor),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Lucebox source\n  repository: {manifest.repository}\n  pin:        {manifest.ref}")
        print(f"  layout:     {manifest.layout}\n  checkout:   {layout.checkout} ({'ready' if checked_out else 'not cloned'})")
        print(f"  runtime:    {manifest.runtime}\n  vendor:     {manifest.vendor.base_commit}")


if __name__ == "__main__":
    main()
