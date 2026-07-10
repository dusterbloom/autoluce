# autoggml v2

Autonomous research harness for **verifiably and reproducibly improving GGML-based inference engines**, starting with [`Luce-Org/lucebox-ggml`](https://github.com/Luce-Org/lucebox-ggml).

## TL;DR

`autoggml` tests small GGML engine changes, rejects correctness or quality regressions,
and keeps only statistically meaningful speedups. It supports local CUDA, HIP, and
Vulkan machines plus leased remote targets such as AMD Strix Halo.

```bash
# Install and run a small local end-to-end experiment.
curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoggml/main/install.sh | bash
cd autoggml
AUTOGGML_BENCHMARKS=smoke uv run autoggml setup
AUTOGGML_BENCHMARKS=smoke uv run autoggml baseline

# Or join an existing team and contribute this machine.
uv run autoggml join --team "$TEAM_URL" --token "$TEAM_TOKEN" --name my-gpu
uv run autoggml worker --once

# Submit a candidate from any connected checkout.
uv run autoggml submit patches/my-candidate.patch --title "My optimization" \
  --backend hip --model deepseek-v4-flash
uv run autoggml status

# Or participate as a research agent.
uv run autoggml agent join --name codex-one --capability implement
uv run autoggml agent next
```

Use `uv run autoggml help` for all commands. On a remotely onboarded Lucebox, the
installed user launcher allows the shorter `autoggml test-drive` form.

## In plain words

If you run an AI model on your own computer, autoggml is a tireless lab assistant for the engine underneath it. It tries one small change at a time, carefully measures whether the model now answers faster — without getting dumber or using more memory — keeps the changes that genuinely help, and undoes the rest. It can even watch how *you* actually use your model (privately, on your own disk) and tune the engine for your real workload instead of a synthetic benchmark.

## What this is

`autoggml` is a self-contained experimentation framework. An AI agent (or a human) proposes a change to `lucebox-ggml`, the harness builds the project, runs a fixed benchmark suite, checks correctness, and records whether the change improved the metric.

The focus of v2 is **verifiability and reproducibility**:

- `autoggml/prepare.py` clones `lucebox-ggml` and pins the current upstream `HEAD` in `work/lucebox-ggml.pin`; every experiment resets to that pin.
- Benchmarks use fixed prompts, seeds, and model configurations.
- The environment (OS, compiler, GPU, dependency versions) is recorded for every run.
- Correctness is checked two ways: generated outputs are compared against golden outputs, and (per benchmark, opt-in) KL divergence against frozen baseline logits catches quality regressions that exact-match can't (`autoggml/bench/kl.py`).
- Results are logged in a machine-readable format with full provenance.
- **Real mode measures every metric or raises** — no silent fallbacks. `--simulate` is the opt-in plumbing mode for CI/no-GPU smoke.
- **Keep/revert is significance-gated**, not raw-improvement, so noise can't drive a random walk.
- `--profile` captures `nsys`/`rocprof` traces and `profiling.classify_bottleneck` reports whether decode is memory-, compute-, or overhead-bound.

The optimization plan — ranked ideas (algorithmic / kernel / graph / runtime), the Strix-Halo unified-memory angle, and execution order — lives in [`ROADMAP.md`](ROADMAP.md).

## Finding your way around

Five things matter; everything else is docs and config:

```
cli.py            Start here: the `autoggml <command>` entry point (routes to autoggml/ modules)
experiment.py     The ONE file you (or the agent) edit to try a change
autoggml/         The engine — read-only during experiments, organized by question:
│  ├── bench/       "Is it better?"      harness, objective (constraints), kl (quality), uncertainty, profiling
│  ├── loop/        "Try it, keep it?"   agent_loop (keep/revert), verify (clean A/B), patches
│  ├── ideation/    "What to try next?"  ideas, selector (rank by bottleneck), propose + llm (optional LLM)
│  ├── parallel/    "Many at once"       runner (fan-out), concurrency (locked leaderboard), worktree
│  ├── prepare.py   Setup: clone, build, download models (GPU auto-detected, GGUFs reused)
│  ├── shadow.py    Shadow bench: benchmark built from your own traffic
│  ├── report.py    Result aggregation and diff
│  └── reproduce.py Reproducibility suite
benchmarks/       What gets measured: fixed prompts, expected outputs, per-benchmark objectives
python_tests/     Proof it works (`uv run pytest -q`)
```

Supporting cast: `program.md` (the autonomous agent's instructions), `ROADMAP.md` (ranked ideas queue), `scripts/` (golden-output generation), `patches/` (patch files used by experiment.py), `install.sh`, `Dockerfile`, `.github/workflows/` (CI). `results.tsv` and `work/` are created by runs, not committed.

## Quick start

### One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoggml/main/install.sh | bash
cd autoggml
AUTOGGML_BENCHMARKS=smoke uv run autoggml setup     # ~1 GB model + build
```

Clones the repo, installs `uv` + dependencies, and runs setup. Want to audit first?
`curl -fsSL <url> -o install.sh && less install.sh`.

Every workflow is available through `uv run autoggml`:

- Team: `join`, `submit`, `status`, `pause`, `resume`, `leave`, `worker`, `coordinator`, `agent`.
- Research: `setup`, `doctor`, `consult`, `freeze`, `baseline`, `run`, `verify`, `profile-report`.
- Supporting tools: `ideas`, `propose`, `harness`, `report`, `reproduce`, `kl-base`, `shadow`, `test-drive`, `onboard`.

## Team workflow

The everyday team interface is `join`, `submit`, `status`, and `worker`.
Coordinator internals, queue labels, manifests, and research-contract YAML stay out of
the normal path.

One team lead starts the restricted coordinator on a machine reachable by the team
(prefer an HTTPS reverse proxy or a private Tailscale address):

```bash
export AUTOGGML_COORDINATOR_TOKEN="$(openssl rand -hex 24)"
uv run autoggml coordinator --listen 127.0.0.1 --port 8765
```

That address is suitable for a same-machine test. For a shared deployment, publish it
through HTTPS or bind it to a private Tailscale address, then give contributors that
reachable URL as `TEAM_URL`.

Each contributor connects once. Hardware and memory are detected automatically; the
explicit flags below are only needed to correct detection:

```bash
uv run autoggml join --team "$TEAM_URL" --token "$TEAM_TOKEN" --name peppi-3090
uv run autoggml status
```

The connection is remembered in `~/.config/autoggml/team.json` with mode `0600`.
Environment variables `AUTOGGML_COORDINATOR_URL` and
`AUTOGGML_COORDINATOR_TOKEN` override that file for managed installations.

Submitters provide a patch and the cells it must pass. The coordinator assigns at most
one active experiment to each physical machine and copies the patch into immutable,
content-addressed storage:

```bash
uv run autoggml submit patches/my-candidate.patch --title "Fuse Sinkhorn" \
  --backend hip --model deepseek-v4-flash
uv run autoggml status
```

On a joined machine, this processes one assigned experiment and returns its result:

```bash
uv run autoggml worker --once
```

`worker` accepts typed candidate data only; the coordinator cannot send arbitrary shell
commands. The worker runs the existing correctness-gated autoggml pipeline under the
host accelerator lock, uses separate CUDA/HIP/Vulkan build directories, and caps builds
at four jobs. `uv run autoggml worker --once --simulate` tests the entire queue lifecycle
without building or using an accelerator. Use `uv run autoggml pause` before taking a
personal machine offline, `uv run autoggml resume` when it is available again, and
`uv run autoggml leave` to remove it from the team.

The file-backed coordinator is deliberately a small deployment unit, not the public
status page. Its service boundary can later be backed by GitHub or the Lucebox control
plane without changing contributor commands.

## Agent challenges

Agents participate as credited researchers, not privileged hardware workers. They
choose bounded tasks, work from a pinned commit in isolated worktrees, and submit only
patches plus structured findings. The existing candidate gate and hardware queue remain
the sole path to accelerator execution.

Create a challenge with distinct approaches so parallel agents explore rather than
produce the same patch repeatedly:

```bash
uv run autoggml agent challenge create \
  --title "Sinkhorn dispatch challenge" \
  --objective "Reduce batch-one Sinkhorn overhead" \
  --why "rocprof attributes 29.4% of decode time to tiny operations" \
  --evidence "capture rp-17" \
  --model deepseek-v4-flash --backend hip --slots 2 \
  --approach "kernel fusion" \
  --approach "persistent buffer reuse"
```

Each agent registers once. Its identity is remembered in
`~/.config/autoggml/agent.json` with mode `0600`:

```bash
uv run autoggml agent join --name codex-kernel-1 --capability implement
uv run autoggml agent next
uv run autoggml agent start <task-id>
```

`start` claims an expiring task lease and creates an isolated worktree at the challenge's
pinned revision. The task packet contains the objective, approach, evidence, expected
impact, difficulty, allowed and forbidden paths, budgets, definition of done, and test
command. Submit the resulting patch and what was learned:

```bash
uv run autoggml agent submit <task-id> --patch candidate.patch \
  --rationale "Fuse the decomposed graph operation" \
  --observation "Removes repeated launches" \
  --risk "May increase register pressure"
```

Implementation submissions remain blind until every implementation has reached a
terminal state and its hardware evaluation finishes. The coordinator then releases the
measured evidence to a reviewer, followed by a recombination task:

```bash
uv run autoggml agent join --name codex-review-1 --capability review
uv run autoggml agent next
uv run autoggml agent join --name codex-hybrid-1 --capability recombine
uv run autoggml agent next
uv run autoggml agent advance <challenge-id>
uv run autoggml agent card <challenge-id>
```

The challenge card ranks measured candidates and retains a contribution graph for
implementers, reviewers, recombiners, and source artifacts. Agent execution failures and
negative hardware results remain durable research evidence; an all-failed round closes
as `inconclusive` rather than hanging. Agent reasoning may run concurrently, while the
existing fleet rule still permits only one active experiment per physical machine.

## Remote machine-aware workflow

Remote targets live in the gitignored `~/.config/autoggml/targets.toml`; start from
[`targets.example.toml`](targets.example.toml). SSH host names and absolute remote paths
do not belong in a research contract or committed result.

One-time onboarding installs a user-local launcher and a local target profile on the
remote host:

```bash
uv run autoggml onboard --target strix-halo
```

After that, the everyday path is deliberately short:

```bash
ssh user@strix-host
autoggml test-drive          # safe: inventory + model/patch check + simulated loop
autoggml test-drive --live   # leased 4K/32-token DeepSeek V4 canary
```

The safe test drive never builds or loads the model. The live canary refuses to start
when another worker holds the lease, accelerator/build activity is visible, or less
than 12 GiB host memory is available. Canary numbers are not admitted to the research
frontier.

The Strix Halo vertical slice is:

```bash
# Lightweight and non-mutating. Writes a stable machine fingerprint when --output is used.
uv run autoggml doctor --target strix-halo --model deepseek-v4-flash --json

# Full model hashing holds the fail-fast GPU lease and aborts if the host is busy.
uv run autoggml consult --target strix-halo --model deepseek-v4-flash \
  --hash-model --output research/contracts/strix-halo-deepseek-v4.yaml

# Each heavy command uses flock -n, refuses <12 GiB available RAM or foreign
# accelerator/build activity, and caps compilation at four jobs.
uv run autoggml setup --target strix-halo --backend hip
uv run autoggml freeze --target strix-halo \
  --contract research/contracts/strix-halo-deepseek-v4.yaml --backend hip
uv run autoggml run --baseline --target strix-halo \
  --contract research/contracts/strix-halo-deepseek-v4.yaml --backend hip
uv run autoggml run --target strix-halo \
  --contract research/contracts/strix-halo-deepseek-v4.yaml --backend hip \
  --experiment-patch deepseek-v4-sinkhorn-77bccaa.patch --profile
uv run autoggml verify --target strix-halo \
  --contract research/contracts/strix-halo-deepseek-v4.yaml --backend hip \
  --experiment-patch deepseek-v4-sinkhorn-77bccaa.patch --rounds 3
```

Use `setup --provision-tools` during an exclusive idle window to install `ccache`
and, for a Vulkan target, `vulkan-tools`, `glslc`, and the Vulkan development headers.

`doctor` reports observations, inferences, unknowns, and current contention separately.
Baseline metrics, quality references, profiler captures, and frontiers are namespaced by
machine fingerprint, model fingerprint, and backend. Changing the model, kernel, ROCm,
or stable machine configuration therefore requires a fresh contract and baseline.

### Manual

You need [uv](https://docs.astral.sh/uv/) installed. It handles Python, the virtual environment, and dependencies in one step. Real builds also require `cmake`, `ccache`, and `ninja-build` (the harness builds with the Ninja generator + ccache launchers for fast incremental rebuilds).

**GPU is auto-detected.** Setup probes for `nvcc` / `hipcc` / `vulkaninfo` / Metal and builds for the best available backend — no `GGML_CUDA=ON` needed. A CPU-only box is refused (its numbers are unrelated to the GPU-bound roadmap); override with `AUTOGGML_ALLOW_CPU=1` for a plumbing build. **Existing GGUFs are reused** — setup scans `~/.cache/huggingface/hub`, LM Studio, `~/models`, etc. before downloading, so you don't re-pull a model you already have. Extend the search path with `AUTOGGML_MODELS=/path/a:/path/b`.

```bash
# 1. Create the virtual environment and install dependencies
uv sync

# 2. No-GPU plumbing smoke (fake measurements; never writes best-score or git state):
uv run pytest -q && uv run autoggml baseline --simulate

# 3. One-time setup: clone lucebox-ggml, download models, build
#    (first build after switching to the Ninja generator requires a clean build dir)
rm -rf work/lucebox-ggml/build
uv run autoggml setup

# 4. Run the baseline benchmark (real mode; raises if unprepared)
uv run autoggml baseline

# 5. Run with the current experiment
uv run autoggml run
```

`uv` creates and manages `.venv/` automatically. Do not create your own virtualenv; `uv run` always uses the project-managed one.

### Fast path: first real result in ~10 minutes

To exercise the full real loop (build → bench → correctness → significance) without the ~35 GB model download, use the tiny smoke model — it downloads only the benchmark(s) you select:

```bash
AUTOGGML_BENCHMARKS=smoke uv run autoggml setup                       # ~1 GB model + build
AUTOGGML_BENCHMARKS=smoke uv run scripts/generate_golden.py --benchmark smoke
AUTOGGML_BENCHMARKS=smoke uv run autoggml baseline            # first real measurement
```

Setup only downloads models referenced by the selected benchmarks, so the orphan `gemma4-26b-a4b` (no benchmark yet) is skipped, and `AUTOGGML_BENCHMARKS=smoke` skips the 27B. Swap the env var back to `qwen36-27b` for the real DFlash benchmark.

### Deterministic container

A `Dockerfile` pins the `uv` version and Python version for CI and local runs. The base image and apt packages are pinned by name but not by digest, so fully byte-for-byte reproducibility requires an additional mirror or digest pin:

```bash
docker build -t autoggml .
docker run --rm -it -v $(pwd)/work:/app/work autoggml
```

## How the autoresearch loop works

1. The agent reads `program.md` and picks the next idea — `uv run autoggml ideas` lists untried `ROADMAP.md` items, and `--bound <memory|compute|overhead>` ranks them by the profiling bottleneck so the high-impact ones come first.
2. *(Optional)* `uv run autoggml propose` asks an OpenAI-compatible LLM for the next experiment given the ranked ideas + current best — see [LLM ideation](#llm-ideation-optional). Disabled unless `OPENAI_BASE_URL` is set.
3. The agent edits `experiment.py` (or calls helpers in `autoggml/loop/patches.py`) to implement one idea.
4. `git commit` the change.
5. `uv run autoggml run` builds `lucebox-ggml` with the experiment applied, runs benchmarks, checks correctness, and either keeps the commit or reverts it. Every shared-state write goes through the file-locked frontier (`concurrency.LockedFrontier`), so workers in parallel can't race.
6. Results are appended to `results.tsv`; the best score **and its stddev** are stored in `.best_score.json`.
7. The commit is kept only if the improvement is **significant** (`--significance`, default `k=1.0`): the score must beat the best by more than `k` times the combined stddev. Otherwise the working tree resets to the previous best.

For searching many ideas at once, `runner.run_parallel` fans experiments out across isolated workers and funnels each result through the locked frontier, and `autoggml/loop/verify.py` does clean A/B verification of candidates before commit — see `program.md`.

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
  (cd .worktrees/w$i && AUTOGGML_FRONTIER="$MAIN" uv run autoggml run) &
done
wait
```

For programmatic fan-out, `runner.run_parallel(specs, run_fn, frontier, max_parallel)`
dispatches specs concurrently and funnels each result through the locked frontier. Supply
your own `run_fn` (local-subprocess / SSH / VM) — dispatch is host-agnostic by design.

## LLM ideation (optional)

The harness is a measurement oracle — it never calls an LLM by default. The LLM is an
*external* coding agent (Claude Code / Codex / Cursor / Aider) that edits `experiment.py`
and runs the loop. Optionally, `autoggml propose` embeds an LLM call for ideation: it ranks the
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
proposal; measurement and keep/revert stay in the loop. Put keys in a gitignored
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

The keep/revert decision is **significance-gated** (`--significance`, default `k=1.0`): a commit is kept only if its score improves on the best by more than `k` times the combined stddev. See `autoggml/bench/uncertainty.py`.

See `autoggml/bench/harness.py` for the exact computation.

## License

Apache-2.0.
