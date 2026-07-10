"""Run the restricted team coordinator API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from autoluce.coordination import FileCoordinationRepository, FleetService
from autoluce.coordinator_http import create_server
from autoluce.agent_challenges import AgentService, CandidatePatchGate, FileAgentRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an authenticated autoluce team coordinator")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.local/share/autoluce/team").expanduser())
    args = parser.parse_args()
    token = os.environ.get("AUTOLUCE_COORDINATOR_TOKEN", "")
    if not token:
        raise SystemExit("Set AUTOLUCE_COORDINATOR_TOKEN before starting the coordinator.")
    service = FleetService(FileCoordinationRepository(args.data_dir))
    agent_service = AgentService(FileAgentRepository(args.data_dir / "agents"), service, CandidatePatchGate())
    server = create_server(
        (args.listen, args.port), service, token=token, upload_dir=args.data_dir / "uploads",
        agent_service=agent_service,
    )
    print(f"autoluce coordinator listening on http://{args.listen}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
