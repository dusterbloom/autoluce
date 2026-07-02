# autoggml v2

Autonomous research harness for improving [`Luce-Org/lucebox-ggml`](https://github.com/Luce-Org/lucebox-ggml).

## What we are optimizing

`lucebox-ggml` is a fork of `llama.cpp` that adds DFlash speculative decoding. The goal of this autoresearch is to discover **verifiable, reproducible improvements** to `lucebox-ggml`:

- Higher decode throughput (tok/s)
- Lower time-to-first-token (TTFT) for prefill
- Higher draft acceptance rate
- Lower memory usage
- Shorter build times
- Correctness preserved (output match against golden outputs, plus optional KL-divergence gate against frozen baseline logits)

## Setup

1. **Agree on a run tag** with the user, e.g., `jun29`.
2. **Create a branch**: `git checkout -b autoggml/<tag>` from `main`.
3. **Read the in-scope files**:
   - `README.md` — project overview.
   - `program.md` — this file.
   - `autoggml/prepare.py` — setup. Do not modify.
   - `autoggml/bench/harness.py` — benchmark harness. Do not modify.
   - `autoggml/loop/agent_loop.py` — keep/revert loop. Do not modify.
   - `autoggml/loop/patches.py` — helpers for common lucebox-ggml modifications. Do not modify; call from `experiment.py`.
   - `autoggml/reproduce.py` — reproducibility suite. Do not modify.
   - `autoggml/report.py` — result aggregation. Do not modify.
   - `experiment.py` — this is what you edit.
4. **Run setup**: `uv run autoggml setup` (one-time; can take 10–30 minutes depending on hardware).
5. **Verify baseline**: `uv run autoggml baseline` should print a score.
6. **Generate golden outputs** after models download: `uv run scripts/generate_golden.py`.
7. **Initialize `results.tsv`** with just the header row (or let the loop create it).

## Experimentation loop

**What you CAN do:**
- Modify `experiment.py` to implement one idea per experiment.
- Add small helper functions inside `experiment.py`.
- Call patch helpers from `autoggml/loop/patches.py` (e.g., `apply_march_native`, `apply_speculative_candidates`).
- Place patch files in `patches/` and reference them from `experiment.py`.

**What you CANNOT do:**
- Modify anything under `autoggml/` (the harness, loop, patches, reproduce, report machinery).
- Install new packages beyond `pyproject.toml`.
- Change the metric or correctness check.
- Commit changes inside the `lucebox-ggml/` submodule.

**The goal is simple: get the highest `score`.** The harness prints:

```
---
score:              134.5000
decode_tok_s:       134.50
prefill_tok_s:      3456.70
acceptance_rate:    0.6234
peak_mem_GiB:       18.2
build_time_s:       245.3
correctness:        pass
```

`score` is the metric to maximize: it equals `decode_tok_s`, but is zeroed by a
correctness failure, a KL-gate failure, or any constraint violation from the
benchmark's `"objective"` block (see README "Metric").

**Correctness is a hard constraint.** If `correctness` is `FAIL`, the experiment is discarded regardless of throughput.

**Simplicity criterion**: All else equal, simpler is better. A tiny throughput gain that adds hundreds of lines of fragile patch code is not worth it.

## Output format

After each run, log to `results.tsv` (tab-separated, 9 columns):

```
commit	score	score_stddev	decode_tok_s	prefill_tok_s	acceptance_rate	peak_mem_GiB	status	description
```

- `commit`: short git hash (7 chars)
- `score`: harness score (0.0000 for crashes)
- `score_stddev`: propagated score uncertainty (0.0000 for crashes)
- `decode_tok_s`: decode throughput (0.0 for crashes)
- `prefill_tok_s`: prefill throughput (0.0 for crashes)
- `acceptance_rate`: speculative acceptance rate (0.0 for crashes)
- `peak_mem_GiB`: peak memory (0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what this experiment tried

## Experiment loop

LOOP FOREVER:

1. Look at the git state and `results.tsv`.
2. Pick the next idea: `uv run autoggml ideas` prints untried `ROADMAP.md` items. If the queue is empty, re-profile (`--profile`), search literature (below), and add ideas.
3. Modify `experiment.py` with one experimental idea.
4. `git commit`.
5. Run the loop: `uv run autoggml run > run.log 2>&1`.
6. Read results: `grep "^score:\|^score_stddev:\|^correctness:" run.log`.
7. If output is empty, the run crashed. Read `tail -n 50 run.log`, attempt a fix.
8. the loop (`autoggml run`) appends to `results.tsv` and keeps or reverts the commit automatically (significance-gated; see `autoggml/bench/uncertainty.py`).
9. If the improvement is significant, the commit is kept; otherwise the loop resets back to the previous best.

For a dry run that does not modify git state, use `uv run autoggml run --dry-run`.
For the baseline, use `uv run autoggml run --baseline` or `uv run autoggml baseline`.

**Timeout**: If the harness takes more than 60 minutes, kill it and treat as a failure.

**Crashes**: If a bug is trivial (typo, wrong path), fix and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the loop begins, continue autonomously until manually interrupted.

## Suggested experiment categories

Start with low-risk, high-leverage changes and measure after each one.

### 1. Build / compile flags
- Try different `CMAKE_BUILD_TYPE` values.
- Enable/disable specific backends (`GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`).
- Try architecture-specific flags (`-march=native`, `-ffast-math`).
- Enable link-time optimization (`-flto`).

### 2. Runtime parameters
- `spec-draft-n-max`, `spec-draft-n-min`, `spec-draft-p-min`.
- KV-cache types for target and draft (`--cache-type-k`, `--cache-type-v`).
- Batch sizes (`-b`, `-ub`).
- GPU layer offloading for draft model (`-ngld`).

### 3. DFlash-specific tuning
- Different `target_layer_ids` configurations (if you can override via metadata or patch).
- Block-size-aware clamping heuristics.
- Draft sampler temperature/top_k for acceptance vs. diversity.

### 4. Code patches (advanced)
- Add early-exit heuristics in the speculative loop.
- Optimize the feature interleaving / cache injection path.
- Reduce synchronization between target and draft contexts.
- Add asynchronous draft generation.

### 5. Reproducibility / harness improvements
These are not experiments, but if you find a bug in the harness, report it to the user instead of silently patching it.

## Literature search

When the ideas queue runs low, mine human knowledge before brainstorming from code context alone. Search in roughly this order of value:

1. **Forks and competitors** — forks claiming better speed carry proven optimizations in their commit history. Adapt them.
2. **Project PRs and issues** — merged "performance" PRs, known bottlenecks, prior attempts.
3. **arXiv / Google Scholar** — papers on optimizing this project or its domain (speculative decoding, KV cache, kernel fusion). Save PDFs to `papers/`.
4. **Technique papers** — general methods (EAGLE/Medusa, operator fusion, cache-oblivious algorithms, lock-free structures).

Rank findings in `ROADMAP.md` (the living, numbered ideas queue; `uv run autoggml ideas` reports what's untried).

**Tagging convention:** when an experiment targets a `ROADMAP.md` item, prefix its description with the item number and cite the source, e.g. `[#3] adaptive K controller (EAGLE-2, Li 2024)`. This lets the ideas tracker track coverage and records the provenance that later goes into a PR body.

## Deterministic reproducibility

Use the provided `Dockerfile` to pin OS, compiler, and `uv` versions:

```bash
docker build -t autoggml .
docker run --rm -it -v $(pwd)/work:/app/work autoggml
```

CI builds this container on every push. Results obtained inside the container are considered the canonical reproducibility target.

## Important constraints

- Do not break the build on the reference hardware.
- Do not change the model weights or tokenizer.
- Do not silently disable correctness checks.
- All patches must be deterministic and reproducible.

## Tips

- Make one change at a time.
- Read the `lucebox-ggml` source before patching (`common/speculative.cpp`, `src/models/dflash.cpp`).
- Use `patches/` for non-trivial changes; keep `experiment.py` as the orchestrator.
- If a change only helps on one benchmark, consider making it conditional.
- Track wall-clock time; some optimizations trade build time for runtime.
