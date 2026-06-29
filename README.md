# autoggml v2

Autonomous research harness for **verifiably and reproducibly improving GGML-based inference engines**, starting with [`Luce-Org/lucebox-ggml`](https://github.com/Luce-Org/lucebox-ggml).

## What this is

`autoggml` is a self-contained experimentation framework. An AI agent (or a human) proposes a change to `lucebox-ggml`, the harness builds the project, runs a fixed benchmark suite, checks correctness, and records whether the change improved the metric.

The focus of v2 is **verifiability and reproducibility**:

- Every experiment starts from a pinned `lucebox-ggml` commit.
- Benchmarks use fixed prompts, seeds, and model configurations.
- The environment (OS, compiler, GPU, dependency versions) is recorded for every run.
- Correctness is checked by comparing generated outputs and perplexity against a baseline.
- Results are logged in a machine-readable format with full provenance.

## Repository layout

```
autoggml/
├── README.md              This file
├── program.md             Instructions for the autonomous agent
├── prepare.py             Read-only setup: clone, build, download models
├── experiment.py          Agent-editable: the change to try
├── harness.py             Read-only benchmark and correctness harness
├── reproduce.py           Read-only reproducibility suite
├── report.py              Read-only result aggregation and diff
├── pyproject.toml         Python dependencies
├── benchmarks/            Benchmark definitions (fixed prompts, expected outputs)
├── patches/               Optional patch files referenced by experiment.py
├── results.tsv            Experiment log (created by the agent, not committed)
└── .github/workflows/     CI that validates the harness on CPU
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
uv run harness.py
```

`uv` creates and manages `.venv/` automatically. Do not create your own virtualenv; `uv run` always uses the project-managed one.

## How the autoresearch loop works

1. The agent reads `program.md`.
2. The agent edits `experiment.py` to implement one idea.
3. `git commit` the change.
4. `uv run harness.py` builds `lucebox-ggml` with the experiment applied, runs benchmarks, and prints a score.
5. If the score improves (higher `tokens_per_second` at equal or better correctness), the commit is kept.
6. Otherwise, `git reset` and try the next idea.

## Metric

The primary metric is **normalized throughput**:

```
score = (decode_tok/s)^2 / (peak_mem_GiB * build_time_s)
```

- Higher is better.
- `decode_tok/s` is squared because speculative decoding cares most about decode throughput.
- Memory and build time act as regularizers to prevent cheating.
- A large correctness penalty is applied if outputs diverge or perplexity regresses.

See `harness.py` for the exact computation.

## License

Apache-2.0.
