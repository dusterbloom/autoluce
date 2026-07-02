# autoggml v2

Autonomous research harness for **verifiably and reproducibly improving GGML-based inference engines**, starting with [`Luce-Org/lucebox-ggml`](https://github.com/Luce-Org/lucebox-ggml).

## In plain words

If you run an AI model on your own computer, autoggml is a tireless lab assistant for the engine underneath it. It tries one small change at a time, carefully measures whether the model now answers faster — without getting dumber or using more memory — keeps the changes that genuinely help, and undoes the rest. It can even watch how *you* actually use your model (privately, on your own disk) and tune the engine for your real workload instead of a synthetic benchmark.

## What this is

`autoggml` is a self-contained experimentation framework. An AI agent (or a human) proposes a change to `lucebox-ggml`, the harness builds the project, runs a fixed benchmark suite, checks correctness, and records whether the change improved the metric.

The focus of v2 is **verifiability and reproducibility**:

- `prepare.py` clones `lucebox-ggml` and pins the current upstream `HEAD` in `work/lucebox-ggml.pin`; every experiment resets to that pin.
- Benchmarks use fixed prompts, seeds, and model configurations.
- The environment (OS, compiler, GPU, dependency versions) is recorded for every run.
- Correctness is checked two ways: generated outputs are compared against golden outputs, and (per benchmark, opt-in) KL divergence against frozen baseline logits catches quality regressions that exact-match can't (`kl.py`).
- Results are logged in a machine-readable format with full provenance.
- **Real mode measures every metric or raises** — no silent fallbacks. `--simulate` is the opt-in plumbing mode for CI/no-GPU smoke.
- **Keep/revert is significance-gated**, not raw-improvement, so noise can't drive a random walk.
- `--profile` captures `nsys`/`rocprof` traces and `profiling.classify_bottleneck` reports whether decode is memory-, compute-, or overhead-bound.

The optimization plan — ranked ideas (algorithmic / kernel / graph / runtime), the Strix-Halo unified-memory angle, and execution order — lives in [`ROADMAP.md`](ROADMAP.md).

## Repository layout

```
autoggml/
├── README.md              This file
├── CHANGELOG.md           Notable changes
├── ROADMAP.md             "Beat llama.cpp" plan: ranked ideas, archive reframe, execution order
├── LICENSE                Apache-2.0
├── program.md             Instructions for the autonomous agent
├── install.sh             One-liner installer (curl | bash)
├── cli.py                 Unified `autoggml <command>` router (subcommands → scripts)
├── prepare.py             Setup: clone, build, download models (GPU auto-detected, GGUFs reused)
├── experiment.py          Agent-editable: the change to try
├── harness.py             Read-only benchmark and correctness harness
├── agent_loop.py          Single-worker keep/revert loop (writes via locked frontier)
├── patches.py             Read-only helpers for common lucebox-ggml patches
├── reproduce.py           Read-only reproducibility suite
├── report.py              Read-only result aggregation and diff
├── uncertainty.py         Score-uncertainty propagation + significance gate
├── objective.py           Constraint checks for the score (k·σ margins, baseline-relative bounds)
├── kl.py                  KL-divergence quality oracle against frozen baseline logits
├── shadow.py              Shadow bench: prompt-capture proxy + benchmark built from your own traffic
├── profiling.py           Backend-aware profiler capture + bottleneck classification
├── selector.py            Rank untried ROADMAP ideas by the active bottleneck
├── ideas.py               Untried-ideas reporter (`--bound` ranks by bottleneck)
├── propose.py             Optional LLM proposer for the next idea (needs OPENAI_BASE_URL)
├── llm.py                 OpenAI-compatible client (cloud or local llama-server/Ollama/vLLM)
├── runner.py              Parallel fan-out: dispatch / screen / run_parallel (live frontier)
├── verify.py              Clean A/B verification before commit
├── concurrency.py         File-locked shared frontier (parallel-safe leaderboard)
├── worktree.py            Git worktree lifecycle (isolated worker trees)
├── Dockerfile             Deterministic container image
├── pyproject.toml         Python dependencies + `autoggml` console script
├── benchmarks/            Benchmark definitions (fixed prompts, expected outputs)
├── patches/               Optional patch files referenced by experiment.py
├── scripts/               Golden-output generation, docker helper
├── results.tsv            Experiment log (created by the agent, not committed)
└── .github/workflows/     CI: lint, pytest, simulation smoke, container build
```

## Quick start

### One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoggml/main/install.sh | bash
cd autoggml
AUTOGGML_BENCHMARKS=smoke uv run autoggml setup     # ~1 GB model + build
```

Clones the repo, installs `uv` + dependencies, and runs setup. Want to audit first?
`curl -fsSL <url> -o install.sh && less install.sh`.

Every script is also a subcommand of `uv run autoggml` (try `uv run autoggml help`):
`setup`, `baseline`, `run`, `ideas`, `propose`, `harness`, `report`, `reproduce`, `kl-base`, `shadow`.

### Manual

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

1. The agent reads `program.md` and picks the next idea — `uv run autoggml ideas` lists untried `ROADMAP.md` items, and `--bound <memory|compute|overhead>` ranks them by the profiling bottleneck so the high-impact ones come first.
2. *(Optional)* `uv run autoggml propose` asks an OpenAI-compatible LLM for the next experiment given the ranked ideas + current best — see [LLM ideation](#llm-ideation-optional). Disabled unless `OPENAI_BASE_URL` is set.
3. The agent edits `experiment.py` (or calls helpers in `patches.py`) to implement one idea.
4. `git commit` the change.
5. `uv run autoggml run` builds `lucebox-ggml` with the experiment applied, runs benchmarks, checks correctness, and either keeps the commit or reverts it. Every shared-state write goes through the file-locked frontier (`concurrency.LockedFrontier`), so workers in parallel can't race.
6. Results are appended to `results.tsv`; the best score **and its stddev** are stored in `.best_score.json`.
7. The commit is kept only if the improvement is **significant** (`--significance`, default `k=1.0`): the score must beat the best by more than `k` times the combined stddev. Otherwise the working tree resets to the previous best.

For searching many ideas at once, `runner.run_parallel` fans experiments out across isolated workers and funnels each result through the locked frontier, and `verify.py` does clean A/B verification of candidates before commit — see `program.md`.

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

## LLM ideation (optional)

The harness is a measurement oracle — it never calls an LLM by default. The LLM is an
*external* coding agent (Claude Code / Codex / Cursor / Aider) that edits `experiment.py`
and runs the loop. Optionally, `propose.py` embeds an LLM call for ideation: it ranks the
untried ideas by the bottleneck and asks the model for one concrete next experiment.

`propose` is gated by `OPENAI_BASE_URL` (no default; disabled = no network call). One
client covers cloud OpenAI and any local OpenAI-compatible backend:

| Backend | `OPENAI_BASE_URL` | Auth |
|---|---|---|
| OpenAI cloud | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| llama.cpp / lucebox `llama-server` | `http://localhost:8080/v1` | none |
| Ollama | `http://localhost:11434/v1` | none |
| vLLM / LM Studio | `http://localhost:8000/v1` / `http://localhost:1234/v1` | none |

```bash
# cloud
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... AUTOGGML_MODEL=gpt-4o-mini \
  uv run autoggml propose --bound memory

# local (the very llama-server this harness builds)
OPENAI_BASE_URL=http://localhost:8080/v1 AUTOGGML_MODEL=<loaded-model> uv run autoggml propose
```

`propose` is ideation only — the agent (or a human) still writes `experiment.py` from the
proposal; measurement and keep/revert stay in `agent_loop.py`. Put keys in a gitignored
`.env`, never in the repo.

## Shadow bench

Standard benchmarks measure a synthetic workload; the shadow bench measures **yours**. A
capture proxy sits in front of your local llama-server and logs the prompts you actually
send; a generated benchmark then scores every candidate change on those prompts, with KL
divergence as the quality gate — so the optimizer speeds up *your* usage and can't trade
away quality on it.

1. Capture: `uv run autoggml shadow proxy --port 8091 --upstream http://127.0.0.1:8080`, then point your client at `:8091` and use the model normally.
2. Build the benchmark: `uv run autoggml shadow build`, then `uv run autoggml kl-base shadow` (freezes the quality reference).
3. Optimize against it: `AUTOGGML_BENCHMARKS=shadow uv run autoggml run`

Privacy: prompts and the KL reference stay on-disk under `~/.autoggml/shadow` and gitignored paths (never committable, never under `benchmarks/`); nothing leaves the box.

## Metric

The objective is **constrained maximization of decode throughput**:

```
score = decode_tok_s   subject to the benchmark's "objective" constraints
```

- Higher is better; a constraint violation zeroes the score exactly like a correctness failure.
- Constraints live in each benchmark JSON, e.g. `{"objective": {"maximize": "decode_tok_s", "constraints": {"peak_mem_GiB": {"max": 22.0}, "prefill_tok_s": {"min_frac_of_baseline": 0.95}}}}`. Each bound must hold with a `k·σ` significance margin (`objective.check_constraints`); relative bounds compare against the baseline metrics persisted by `autoggml baseline` (`work/baseline_metrics.json`).
- Decode and prefill throughput (and their stddev) are parsed from `llama-bench`.
- `peak_mem_GiB` is measured via `/usr/bin/time -v` (a missing profiler/source raises — no silent fallback).
- `acceptance_rate` is a logged diagnostic (not scored): speculative runs that don't report it raise; non-speculative runs simply omit it.
- `build_time_s` is measured and reported but **not** scored: with ccache + a preserved build dir it is cache-state-dependent, so scoring it would make runs non-reproducible.
- A correctness failure forces the score to 0.

The keep/revert decision is **significance-gated** (`--significance`, default `k=1.0`): a commit is kept only if its score improves on the best by more than `k` times the combined stddev. See `uncertainty.py`.

See `harness.py` for the exact computation.

## License

Apache-2.0.
