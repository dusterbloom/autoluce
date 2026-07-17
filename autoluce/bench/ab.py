"""`autoluce ab`: interleaved ABBA A/B measurement on one machine.

Launches the clean and candidate dflash_server arms in mirrored ABBA blocks on
a free local port, measures each activation fresh (new process, warmup excluded),
and decides with a paired one-sample t-test on block deltas. This is the local
primitive behind the "interleaved clean/candidate measurements on the same
machine session" frontier rule: sequential or historical comparisons are
diagnostic only, because clocks, thermals, and neighbour load move results by
several percent.

Metric note: this driver times the full chat-completion request client-side
(output tokens / wall second), because some product backends do not populate
usage.timings. The measurement source is recorded as WALL_CLOCK_SOURCE so the
evidence is never mislabeled as server-reported timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from autoluce.bench.interleave import CANDIDATE, CLEAN, run_interleaved
from autoluce.runtime.dflash_http import (
    DflashHttpClient,
    DflashServer,
    _free_port,
    build_server_command,
    product_environment_overrides,
)

WALL_CLOCK_SOURCE = "wall_clock.chat_completion"

DEFAULT_PROMPTS = [
    "Write a Python function to compute the factorial of a number.",
    "Explain virtual memory in two concise sentences.",
]


def _complete_wallclock(client: DflashHttpClient, prompt: str, max_tokens: int) -> tuple[int, float]:
    """One non-streaming completion; returns (completion_tokens, elapsed_seconds).

    Mirrors DflashHttpClient.complete's request body, but parses the response
    leniently: backends that leave usage.timings zeroed are still measurable
    via wall clock, which is exactly the case this driver exists for.
    """
    body = {
        "model": "autoluce-ab",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "prefix_cache": {"scope": "off"},
    }
    start = time.perf_counter()
    response = client.session.post(
        f"{client.base_url}/v1/chat/completions", json=body, timeout=client.timeout_s
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    usage = response.json().get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens <= 0:
        raise RuntimeError(f"completion returned {completion_tokens} tokens; cannot score wall clock")
    return completion_tokens, elapsed


def _parse_assignments(items: list[str] | None, kind: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--{kind} expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        parsed[name] = value
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_spec(args: argparse.Namespace, label: str) -> dict[str, Any]:
    binary = getattr(args, f"{label}_binary") or args.binary
    if binary is None or not Path(binary).exists():
        raise RuntimeError(f"{label} binary not found; pass --binary or --{label}-binary")
    flags = _parse_assignments(getattr(args, f"{label}_flags"), f"{label}-flag")
    env = product_environment_overrides(_parse_assignments(getattr(args, f"{label}_env"), f"{label}-env"))
    return {"binary": Path(binary), "flags": flags, "env": env}


def _measure_arm(label: str, spec: dict[str, Any], args: argparse.Namespace, prompts: list[str]) -> float:
    """One fresh activation: launch, warm up (excluded), measure repetitions, tear down."""
    port = _free_port()
    command = build_server_command(
        spec["binary"], Path(args.model), Path(args.draft) if args.draft else None,
        "127.0.0.1", port, args.max_ctx, spec["flags"],
    )
    with DflashServer(command, port=port, runtime_env=spec["env"]) as client:
        _complete_wallclock(client, prompts[0], args.max_tokens)  # warmup: graph/allocation work excluded
        values = []
        for index in range(args.repetitions):
            tokens, elapsed = _complete_wallclock(client, prompts[index % len(prompts)], args.max_tokens)
            values.append(tokens / elapsed)
        return fmean(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", help="dflash_server binary shared by both arms")
    parser.add_argument("--clean-binary", help="override the clean arm's binary")
    parser.add_argument("--candidate-binary", help="override the candidate arm's binary")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--draft", help="optional draft model path")
    parser.add_argument("--max-ctx", type=int, default=8192)
    parser.add_argument("--prompt", action="append", dest="prompts", help="benchmark prompt (repeatable)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3, help="measured requests per activation")
    parser.add_argument("--blocks", type=int, default=8, help="ABBA blocks (4 activations each)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--clean-flag", action="append", dest="clean_flags", help="clean-arm server flag NAME=VALUE")
    parser.add_argument("--candidate-flag", action="append", dest="candidate_flags", help="candidate-arm server flag NAME=VALUE")
    parser.add_argument("--clean-env", action="append", dest="clean_env", help="clean-arm product env NAME=VALUE")
    parser.add_argument("--candidate-env", action="append", dest="candidate_env", help="candidate-arm product env NAME=VALUE")
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    args = parser.parse_args(argv)

    prompts = args.prompts or DEFAULT_PROMPTS
    specs = {label: _arm_spec(args, label) for label in (CLEAN, CANDIDATE)}
    started = time.time()

    def progress(step: int, total: int, label: str, value: float) -> None:
        print(f"[{step:>2}/{total}] {label:<9} {value:9.2f} tok/s", flush=True)

    result = run_interleaved(
        lambda label: _measure_arm(label, specs[label], args, prompts),
        blocks=args.blocks,
        metric="output_tok_s_wallclock",
        alpha=args.alpha,
        on_progress=None if args.json else progress,
    )

    report = {
        "measurement_source": WALL_CLOCK_SOURCE,
        "metric": result.metric,
        "model": str(args.model),
        "binaries": {label: {"path": str(spec["binary"]), "sha256": _sha256(spec["binary"])} for label, spec in specs.items()},
        "flags": {label: spec["flags"] for label, spec in specs.items()},
        "env": {label: spec["env"] for label, spec in specs.items()},
        "workload": {"prompts": len(prompts), "max_tokens": args.max_tokens, "repetitions": args.repetitions},
        "alpha": args.alpha,
        "duration_s": round(time.time() - started, 1),
        "result": {**asdict(result), "sequence": [list(item) for item in result.sequence]},
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if result.significant:
            verdict = "candidate IMPROVES"
        elif result.regression:
            verdict = "candidate REGRESSES"
        else:
            verdict = "no significant difference"
        print(f"\nclean {result.clean_mean:.2f} vs candidate {result.candidate_mean:.2f} tok/s "
              f"-> effect {result.effect:+.2f} ({result.effect_pct:+.2f}%), "
              f"t({result.test.df:.1f})={result.test.statistic:.2f}, p={result.test.p_value:.4f}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
