"""Remote A/B entry point for the Lucebox product runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoluce.contracts import ResearchContract
from autoluce.source_layout import SourceLayout
from autoluce.targets import TargetConfig


def verify(
    target: TargetConfig,
    contract: ResearchContract,
    patch: str,
    backend: str,
    rounds: int,
    k: float,
) -> dict:
    """Reject until interleaved A/B scheduling is ported to the HTTP runtime."""
    del target, contract, patch, backend, rounds, k
    SourceLayout.resolve().require_capability("product-benchmark")
    raise RuntimeError("interleaved remote A/B verification is not yet wired to dflash_server")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interleaved remote A/B verification")
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--experiment-patch", required=True)
    parser.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--significance", type=float, default=1.0)
    args = parser.parse_args()
    verify(
        TargetConfig.load(args.target),
        ResearchContract.read(args.contract),
        args.experiment_patch,
        args.backend,
        args.rounds,
        args.significance,
    )


if __name__ == "__main__":
    main()
