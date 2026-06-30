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

## Repository layout

```
autoggml/
├── README.md              This file
├── program.md             Instructions for the autonomous agent
├── prepare.py             Read-only setup: clone, build, download models
├── experiment.py          Agent-editable: the change to try
├── harness.py             Read-only benchmark and correctness harness
├── agent_loop.py          Read-only keep/revert experiment loop
├── patches.py             Read-only helpers for common lucebox-ggml patches
├── reproduce.py           Read-only reproducibility suite
├── report.py              Read-only result aggregation and diff
├── Dockerfile             Deterministic container image
├── pyproject.toml         Python dependencies
├── benchmarks/            Benchmark definitions (fixed prompts, expected outputs)
├── patches/               Optional patch files referenced by experiment.py
├── results.tsv            Experiment log (created by the agent, not committed)
└── .github/workflows/     CI that validates the harness and container
```

## Quick start

You need [uv](https://docs.astral.sh/uv/) installed. It handles Python, the virtual environment, and dependencies in one step.

```bash
# 1. Create the virtual environment and install dependencies
uv sync

# 2. One-time setup: clone lucebox-ggml, download models, build
uv run prepare.py

# 3. Run the baseline benchmark
uv run harness.py --baseline

# 4. Run with the current experiment
uv run agent_loop.py
```

`uv` creates and manages `.venv/` automatically. Do not create your own virtualenv; `uv run` always uses the project-managed one.

### Deterministic container

A `Dockerfile` pins the `uv` version and Python version for CI and local runs. The base image and apt packages are pinned by name but not by digest, so fully byte-for-byte reproducibility requires an additional mirror or digest pin:

```bash
docker build -t autoggml .
docker run --rm -it -v $(pwd)/work:/app/work autoggml
```

## How the autoresearch loop works

1. The agent reads `program.md`.
2. The agent edits `experiment.py` (or calls helpers in `patches.py`) to implement one idea.
3. `git commit` the change.
4. `uv run agent_loop.py` builds `lucebox-ggml` with the experiment applied, runs benchmarks, checks correctness, and either keeps the commit or reverts it.
5. Results are appended to `results.tsv` and the best score is stored in `.best_score.json`.
6. If the score improves, the commit is kept; otherwise the working tree is reset to the previous best.

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
