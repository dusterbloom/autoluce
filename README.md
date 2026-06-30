# autoggml v2

Autonomous research harness for **verifiably and reproducibly improving GGML-based inference engines**, starting with [`Luce-Org/lucebox-ggml`](https://github.com/Luce-Org/lucebox-ggml).

## What this is

`autoggml` is a self-contained experimentation framework. An AI agent (or a human) proposes a change to `lucebox-ggml`, the harness builds the project, runs a fixed benchmark suite, checks correctness, and records whether the change improved the metric.

The focus of v2 is **verifiability and reproducibility**:

- `prepare.py` clones `lucebox-ggml` and pins the current upstream `HEAD` in `work/lucebox-ggml.pin`; every experiment resets to that pin.
- Benchmarks use fixed prompts, seeds, and model configurations.
- The environment (OS, compiler, GPU, dependency versions) is recorded for every run.
- Correctness is checked by comparing generated outputs against golden outputs (perplexity is not yet implemented).
- Results are logged in a machine-readable format with full provenance.
- **Real mode measures every metric or raises** — no silent fallbacks. `--simulate` is the opt-in plumbing mode for CI/no-GPU smoke.
- **Keep/revert is significance-gated**, not raw-improvement, so noise can't drive a random walk.
- `--profile` captures `nsys`/`rocprof` traces and `profiling.classify_bottleneck` reports whether decode is memory-, compute-, or overhead-bound.

The optimization plan — ranked ideas (algorithmic / kernel / graph / runtime), the Strix-Halo unified-memory angle, and execution order — lives in [`ROADMAP.md`](ROADMAP.md).

## Repository layout

```
autoggml/
├── README.md              This file
├── ROADMAP.md             "Beat llama.cpp" plan: ranked ideas, execution order
├── CHANGELOG.md           Notable changes
├── LICENSE                Apache-2.0
├── program.md             Instructions for the autonomous agent
├── prepare.py             Read-only setup: clone, build, download models
├── experiment.py          Agent-editable: the change to try
├── harness.py             Read-only benchmark and correctness harness
├── agent_loop.py          Single-worker keep/revert loop (writes via locked frontier)
├── patches.py             Read-only helpers for common lucebox-ggml patches
├── reproduce.py           Read-only reproducibility suite
├── report.py              Read-only result aggregation and diff
├── uncertainty.py         Score-uncertainty propagation + significance gate
├── profiling.py           Backend-aware profiler capture + bottleneck classification
├── runner.py              Parallel fan-out: dispatch / screen / run_parallel (live frontier)
├── verify.py              Clean A/B verification before commit
├── concurrency.py         File-locked shared frontier (parallel-safe leaderboard)
├── worktree.py            Git worktree lifecycle (isolated worker trees)
├── ideas.py               Untried-ideas reporter for ROADMAP.md
├── Dockerfile             Deterministic container image
├── pyproject.toml         Python dependencies
├── benchmarks/            Benchmark definitions (fixed prompts, expected outputs)
├── patches/               Optional patch files referenced by experiment.py
├── scripts/               Golden-output generation, docker helper
├── results.tsv            Experiment log (created by the agent, not committed)
└── .github/workflows/     CI: lint, pytest, simulation smoke, container build
```

## Quick start

You need [uv](https://docs.astral.sh/uv/) installed. It handles Python, the virtual environment, and dependencies in one step. Real builds also require `cmake`, `ccache`, and `ninja-build` (the harness builds with the Ninja generator + ccache launchers for fast incremental rebuilds).

**GPU is auto-detected.** `prepare.py` probes for `nvcc` / `hipcc` / `vulkaninfo` / Metal and builds for the best available backend — no `GGML_CUDA=ON` needed. A CPU-only box is refused (its numbers are unrelated to the GPU-bound roadmap); override with `AUTOGGML_ALLOW_CPU=1` for a plumbing build. **Existing GGUFs are reused** — `prepare.py` scans `~/.cache/huggingface/hub`, LM Studio, `~/models`, etc. before downloading, so you don't re-pull a model you already have. Extend the search path with `AUTOGGML_MODELS=/path/a:/path/b`.

```bash
# 1. Create the virtual environment and install dependencies
uv sync

# 2. No-GPU plumbing smoke (fake measurements; never writes best-score or git state):
uv run pytest -q && uv run harness.py --baseline --simulate

# 3. One-time setup: clone lucebox-ggml, download models, build
#    (first build after switching to the Ninja generator requires a clean build dir)
rm -rf work/lucebox-ggml/build
uv run prepare.py

# 4. Run the baseline benchmark (real mode; raises if unprepared)
uv run harness.py --baseline

# 5. Run with the current experiment
uv run agent_loop.py
```

`uv` creates and manages `.venv/` automatically. Do not create your own virtualenv; `uv run` always uses the project-managed one.

### Fast path: first real result in ~10 minutes

To exercise the full real loop (build → bench → correctness → significance) without the ~35 GB model download, use the tiny smoke model — it downloads only the benchmark(s) you select:

```bash
AUTOGGML_BENCHMARKS=smoke uv run prepare.py                       # ~1 GB model + build
AUTOGGML_BENCHMARKS=smoke uv run scripts/generate_golden.py --benchmark smoke
AUTOGGML_BENCHMARKS=smoke uv run harness.py --baseline            # first real measurement
```

`prepare.py` only downloads models referenced by the selected benchmarks, so the orphan `gemma4-26b-a4b` (no benchmark yet) is skipped, and `AUTOGGML_BENCHMARKS=smoke` skips the 27B. Swap the env var back to `qwen36-27b` for the real DFlash benchmark.

### Deterministic container

A `Dockerfile` pins the `uv` version and Python version for CI and local runs. The base image and apt packages are pinned by name but not by digest, so fully byte-for-byte reproducibility requires an additional mirror or digest pin:

```bash
docker build -t autoggml .
docker run --rm -it -v $(pwd)/work:/app/work autoggml
```

## How the autoresearch loop works

1. The agent reads `program.md` and picks the next idea (`uv run ideas.py` lists untried `ROADMAP.md` items).
2. The agent edits `experiment.py` (or calls helpers in `patches.py`) to implement one idea.
3. `git commit` the change.
4. `uv run agent_loop.py` builds `lucebox-ggml` with the experiment applied, runs benchmarks, checks correctness, and either keeps the commit or reverts it.
5. Results are appended to `results.tsv`; the best score **and its stddev** are stored in `.best_score.json`.
6. The commit is kept only if the improvement is **significant** (`--significance`, default `k=1.0`): the score must beat the best by more than `k` times the combined stddev. Otherwise the working tree resets to the previous best.

For searching many ideas at once, `runner.py` runs experiments in parallel (screen) and `verify.py` does clean A/B verification of candidates before commit — see `program.md`.

## Parallel runs

The shared leaderboard is safe under concurrency. `agent_loop` writes `.best_score.json`
and `results.tsv` through a file lock (`concurrency.LockedFrontier`) and re-verifies each
candidate against the **live** frontier, not the snapshot it started with — so a worker can
no longer "keep" a win a faster sibling already beat. To run workers in parallel on one
host, give each its own git worktree (isolated `build/` dir) and point them all at one
shared frontier:

```bash
MAIN=$(pwd)
for i in 1 2 3 4; do
  uv run python -c "from pathlib import Path; from worktree import ensure_worktree; ensure_worktree(Path('.'), 'w$i')"
  (cd .worktrees/w$i && AUTOGGML_FRONTIER="$MAIN" uv run agent_loop.py) &
done
wait
```

For programmatic fan-out, `runner.run_parallel(specs, run_fn, frontier, max_parallel)`
dispatches specs concurrently and funnels each result through the locked frontier. Supply
your own `run_fn` (local-subprocess / SSH / VM) — dispatch is host-agnostic by design.

## Metric

The primary metric is **throughput per unit memory**:

```
score = (decode_tok/s * prefill_tok/s * acceptance_rate) / peak_mem_GiB
```

- Higher is better.
- Decode and prefill throughput (and their stddev) are parsed from `llama-bench`.
- `peak_mem_GiB` is measured via `/usr/bin/time -v` (a missing profiler/source raises — no silent fallback).
- `acceptance_rate` is parsed when the benchmark reports it; for speculative runs that don't report it, the neutral value `1.0` is used (identical for baseline and experiment, so comparisons stay fair). See `ROADMAP.md` for closing this gap.
- `build_time_s` is measured and reported but **not** scored: with ccache + a preserved build dir it is cache-state-dependent, so scoring it would make runs non-reproducible.
- A correctness failure forces the score to 0.

The keep/revert decision is **significance-gated** (`--significance`, default `k=1.0`): a commit is kept only if its score improves on the best by more than `k` times the combined stddev. See `uncertainty.py`.

See `harness.py` for the exact computation.

## License

Apache-2.0.
