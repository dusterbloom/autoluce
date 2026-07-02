"""
KL-to-baseline quality oracle (Tier-1) for autoggml v2.

Reuses llama.cpp's built-in KL tooling (llama-perplexity); no custom logit
storage. Verified against work/lucebox-ggml (common/arg.cpp,
tools/perplexity/perplexity.cpp):

    save base logits:  llama-perplexity -m MODEL -f TEXT --kl-divergence-base BASE
    compute KL:        llama-perplexity -m MODEL --kl-divergence-base BASE --kl-divergence

The check run takes no -f: the evaluation tokens are embedded in the base file.
Summary lines parsed: 'Mean    KLD:', 'Maximum KLD:', '99.9%   KLD:'.

CRITICAL DESIGN INVARIANT: the KL reference is generated ONCE from the ORIGINAL
baseline engine build and never regenerated from the rolling best — quality
drift cannot compound. generate_kl_base refuses to overwrite an existing
reference, and `autoggml kl-base` resets lucebox-ggml to the pinned baseline
before building. The reference lives next to the golden outputs
(benchmarks/golden/<benchmark>.kl_base.bin).

Pure logic (command builders, parsing, gating) is separated from the IO
wrappers so every edge case is testable with synthetic data.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from autoggml import ROOT
BENCHMARKS_DIR = ROOT / "benchmarks"
BUILD_DIR = ROOT / "work" / "lucebox-ggml" / "build"

DEFAULT_KL_TAU = 0.01


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def kl_base_path(benchmark_name: str) -> Path:
    """Reference logits live next to the golden outputs, keyed by benchmark name."""
    return BENCHMARKS_DIR / "golden" / f"{benchmark_name}.kl_base.bin"


def build_kl_base_cmd(perplexity_bin, model, text_file, base_file, extra: list[str] | None = None) -> list[str]:
    """llama-perplexity invocation that saves reference logits from a normal run."""
    return [
        str(perplexity_bin), "-m", str(model), "-f", str(text_file),
        "--kl-divergence-base", str(base_file), *(extra or []),
    ]


def build_kl_check_cmd(perplexity_bin, model, base_file, extra: list[str] | None = None) -> list[str]:
    """llama-perplexity invocation that computes KL against saved reference logits.

    No -f: the evaluation tokens are read back from the base file itself."""
    return [
        str(perplexity_bin), "-m", str(model),
        "--kl-divergence-base", str(base_file), "--kl-divergence", *(extra or []),
    ]


_FLOAT = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_KL_PATTERNS = {
    "mean_kld": re.compile(r"Mean\s+KLD:\s*" + _FLOAT),
    "max_kld": re.compile(r"Maximum\s+KLD:\s*" + _FLOAT),
    "p999_kld": re.compile(r"99\.9%\s+KLD:\s*" + _FLOAT),
}


_KL_GATED = ("mean_kld", "max_kld")  # p999_kld is diagnostic-only


def parse_kl_output(stdout: str) -> dict[str, float]:
    """Parse the KLD summary block. Gated metrics: measure or raise — no silent defaults."""
    kl: dict[str, float] = {}
    for name, pattern in _KL_PATTERNS.items():
        m = pattern.search(stdout)
        if not m:
            if name in _KL_GATED:
                raise RuntimeError(f"could not parse {name} from llama-perplexity --kl-divergence output")
            continue
        kl[name] = float(m.group(1))
    return kl


def kl_gate(kl: dict, tau: float = DEFAULT_KL_TAU) -> list[str]:
    """Quality gate: mean_kld <= tau and max_kld <= 10*tau. Returns violations."""
    violations: list[str] = []
    if kl["mean_kld"] > tau:
        violations.append(f"mean_kld: {kl['mean_kld']:.6g} > tau {tau:.6g}")
    if kl["max_kld"] > 10 * tau:
        violations.append(f"max_kld: {kl['max_kld']:.6g} > 10*tau {10 * tau:.6g}")
    return violations


def resolve_kl_tau(spec: dict) -> float:
    """tau comes from the benchmark JSON "objective" block key "kl_tau" (default 0.01)."""
    return spec.get("objective", {}).get("kl_tau", DEFAULT_KL_TAU)


# ---------------------------------------------------------------------------
# IO wrappers (same subprocess conventions as harness.run)
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, timeout: int | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout, env=env)


def _perplexity_bin() -> Path:
    binary = BUILD_DIR / "bin" / "llama-perplexity"
    if not binary.exists():
        raise RuntimeError(f"{binary} not found; run prepare.py / the harness build first")
    return binary


def generate_kl_base(benchmark_name: str, model, text_file, extra: list[str] | None = None, timeout: int = 3600) -> Path:
    """Save reference logits for a benchmark. Refuses to overwrite (see invariant)."""
    base = kl_base_path(benchmark_name)
    if base.exists():
        raise RuntimeError(
            f"{base} already exists; the KL reference is generated once from the original "
            "baseline build and never regenerated — delete it manually only if the baseline itself changed"
        )
    if not Path(text_file).exists():
        raise RuntimeError(f"kl_text file {text_file} not found")
    base.parent.mkdir(parents=True, exist_ok=True)
    run(build_kl_base_cmd(_perplexity_bin(), model, text_file, base, extra), timeout=timeout)
    if not base.exists():
        raise RuntimeError(f"llama-perplexity did not write {base}")
    return base


def check_kl(benchmark_name: str, model, extra: list[str] | None = None, timeout: int = 3600) -> dict[str, float]:
    """Compute KL of the current build against the frozen baseline reference."""
    base = kl_base_path(benchmark_name)
    if not base.exists():
        raise RuntimeError(
            f"benchmark '{benchmark_name}' declares kl_text but the KL reference {base} is missing; "
            f"generate it from the baseline build with `uv run autoggml kl-base {benchmark_name}`"
        )
    proc = run(build_kl_check_cmd(_perplexity_bin(), model, base, extra), timeout=timeout)
    return parse_kl_output(proc.stdout + "\n" + proc.stderr)


# ---------------------------------------------------------------------------
# CLI entry: generate the reference for a named benchmark
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate the KL reference logits for a benchmark from the ORIGINAL baseline engine build")
    parser.add_argument("benchmark", help="benchmark name (benchmarks/<name>.json)")
    args = parser.parse_args()

    # Lazy imports: harness imports kl at module level.
    from autoggml.bench import harness
    from experiment import reset_lucebox

    spec_path = BENCHMARKS_DIR / f"{args.benchmark}.json"
    if not spec_path.exists():
        print(f"kl-base: unknown benchmark '{args.benchmark}'", file=sys.stderr)
        sys.exit(2)
    spec = json.loads(spec_path.read_text())
    kl_text = spec.get("kl_text")
    if not kl_text:
        print(f"kl-base: benchmark '{args.benchmark}' declares no 'kl_text'; nothing to do", file=sys.stderr)
        sys.exit(2)

    reset_lucebox()  # the reference must come from the pristine baseline tree
    harness.build()
    entry = spec.get("manifest_entry", args.benchmark)
    model = harness.resolve_model(entry, "target")
    if model is None or not model.exists():
        raise RuntimeError(f"target model for benchmark '{args.benchmark}' not found; run prepare.py")
    base = generate_kl_base(args.benchmark, model, ROOT / kl_text)
    print(f"KL reference written: {base}")


if __name__ == "__main__":
    main()
