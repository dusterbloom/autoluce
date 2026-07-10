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
    """Reject live verification until the HTTP product adapter owns the workflow.

    The previous implementation drove standalone llama.cpp binaries and backend-op
    tests that are not part of the vendored Lucebox product. Keeping that script
    behind a nominal feature flag would make enabling the new adapter unsafe.
    """
    del target, contract, patch, backend, rounds, k
    SourceLayout.resolve().require_capability("product-benchmark")
    raise RuntimeError("the Lucebox product benchmark adapter is not registered")


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
