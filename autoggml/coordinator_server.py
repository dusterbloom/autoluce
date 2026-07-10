"""Run the restricted team coordinator API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from autoggml.coordination import FileCoordinationRepository, FleetService
from autoggml.coordinator_http import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an authenticated autoggml team coordinator")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/autoggml/team").expanduser())
    args = parser.parse_args()
    token = os.environ.get("AUTOGGML_COORDINATOR_TOKEN", "")
    if not token:
        raise SystemExit("Set AUTOGGML_COORDINATOR_TOKEN before starting the coordinator.")
    service = FleetService(FileCoordinationRepository(args.data_dir))
    server = create_server((args.listen, args.port), service, token=token, upload_dir=args.data_dir / "uploads")
    print(f"autoggml coordinator listening on http://{args.listen}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
