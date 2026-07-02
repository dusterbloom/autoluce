# Changelog

All notable changes to autoggml are documented here. The project has not tagged
a release yet; everything currently lives under `[Unreleased]`.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **KL quality oracle** (`kl.py`, `autoggml kl-base <benchmark>`): candidates are checked against frozen reference logits via llama.cpp's built-in `--kl-divergence`; a gate violation (`mean_kld > tau` or `max_kld > 10·tau`, `tau` per benchmark via `objective.kl_tau`, default 0.01) zeroes the score like a correctness failure. The reference is generated **once** from the pinned baseline build and never regenerated, so quality drift cannot compound. Opt-in per benchmark via `"kl_text"`.
- **Shadow bench** (`shadow.py`, `autoggml shadow proxy|build`): optimize for your own traffic. A stdlib capture proxy in front of your local `llama-server` tees prompts to `~/.autoggml/shadow/prompts.jsonl`; `shadow build` turns the last day's deduplicated prompts into a `shadow` benchmark whose quality gate is KL divergence on those very prompts. Prompt files and the KL reference are gitignored — nothing leaves the machine.
- Unified `autoggml` CLI (`cli.py`): `uv run autoggml <setup|baseline|run|ideas|propose|harness|report|reproduce>` routes to the existing scripts (reimplements nothing). One-liner `install.sh` (`curl | bash`) clones, ensures `uv`, syncs, prints next steps.
- GPU auto-detection: `prepare.py`/`harness.py` probe for `nvcc`/`hipcc`/`vulkaninfo`/Metal and build for the best backend by default — no `GGML_CUDA=ON` needed. CPU-only is refused unless `AUTOGGML_ALLOW_CPU=1`. Existing GGUFs are reused from the HF cache / LM Studio / `~/models` (`AUTOGGML_MODELS` extends the search) before downloading.
- Parallel-safe shared leaderboard (`concurrency.LockedFrontier`): file-locked `.best_score.json` + `results.tsv`; `claim_best_if_significant` re-verifies against the **live** frontier under the lock, so concurrent workers can't keep a stale-snapshot win. `worktree.py` isolates per-worker trees; `runner.run_parallel` fans out and funnels through the frontier; `agent_loop` honors `AUTOGGML_FRONTIER` so worktree workers share one leaderboard.
- Profile-driven ideation (`selector.rank_by_bottleneck`): `ideas --bound <memory|compute|overhead>` ranks untried `ROADMAP.md` items so those targeting the active bottleneck come first.
- Optional embedded LLM (`llm.py` + `propose.py`): one OpenAI-compatible client for cloud or local (`llama-server`/Ollama/vLLM/LM Studio), env-gated by `OPENAI_BASE_URL` (disabled by default; never fires silently). `propose` asks the model for the next experiment given the ranked ideas + current best.
- `ROADMAP.md` "Meta" section (#15–18): the machine × workload grid reframed as a quality-diversity archive (one elite per cell) with novelty + recombination (#18) as the operator that moves beyond hill-climbing — grounded in AutoKernel (`arXiv:2603.21331`) / AlphaEvolve prior art.
- Significance-gated scoring: keep/revert only on improvements exceeding `k·σ`
  (`--significance`, default `k=1.0`). Score uncertainty propagated via
  relative-error over the multiplicative metric (`uncertainty.py`).
- No-fabrication real mode: every metric is measured or the run raises; `--simulate`
  opt-in for plumbing/CI (never writes best-score or git state).
- `peak_mem_GiB` measured via `/usr/bin/time -v`.
- Correctness generation extracted via the `llama_print_timings:` delimiter, shared
  by the harness and the golden generator (`scripts/generate_golden.py`) so the two
  cannot drift.
- `--profile`: backend-aware profiler capture (`nsys`/`rocprof`) per benchmark to
  `results/profiles/`, plus `classify_bottleneck` (memory/compute/overhead verdict
  pointing at `ROADMAP.md` items).
- ccache + Ninja generator in both build sites and the Dockerfile for fast
  incremental rebuilds.
- Parallel experiment fan-out (`runner.py`): `dispatch` + `screen` over an injected
  `run_fn` (local subprocess / SSH / `sky exec` workers are interchangeable).
- Clean A/B verification gate (`verify.py`): re-measure optimized and baseline
  clean-built on one quiet worker before committing.
- Living ideas queue (`ideas.py`): `uv run ideas.py` reports untried `ROADMAP.md`
  items; `[#N] (paper: X)` tagging convention for coverage + PR provenance.
- `ROADMAP.md`: the "beat llama.cpp" plan (measure-first, ranked algorithmic/kernel/
  graph ideas, Strix-Halo UMA moat, execution order).
- `pytest` runs in CI.

### Changed
- **Constrained objective replaces the product score**: `score = decode_tok_s` only; `peak_mem_GiB` / `prefill_tok_s` are now constraints declared per benchmark in an `"objective"` block (`objective.check_constraints`, k·σ significance margin; `min_frac_of_baseline` compares against `work/baseline_metrics.json` persisted by `autoggml baseline`). A violation zeroes the score exactly like a correctness failure. Speculative runs that don't report `acceptance_rate` now **raise** (the neutral-1.0 fallback is gone); acceptance stays as a logged diagnostic only. **Existing `.best_score.json` / baselines are invalid — re-measure.**
- `pyproject.toml` gains `[build-system]` (setuptools) + `py-modules`, so `uv` installs the project and the `autoggml` console script is generated (previously `[project.scripts]` was silently ignored — no `[build-system]` meant a virtual project).
- `agent_loop.py` routes all four outcomes (baseline/keep/discard/crash) through `LockedFrontier`; the old direct file writes (`load_best*`/`save_best_score`/`log_result`) are removed.
- `profiling.ROADMAP_FOR_BOUND` and `ideas.descriptions_from_results` are now public (shared by `selector`/`propose`).
- Keep/revert decision is now significance-gated (was raw `score > best`).
- `build_time_s` is measured and reported but never scored: with ccache + a
  preserved build dir it is cache-state-dependent and would make runs
  non-reproducible.
- `reset_lucebox` preserves `work/lucebox-ggml/build` (`git clean -fd -e build`) so
  the loop gets incremental rebuilds instead of a full rebuild every run.
- `.best_score.json` and `results.tsv` persist `score_stddev` (schema widened).
- Golden generator uses the harness's shared command builder + text extraction.

### Fixed
- `experiment.py` no longer crashes on `apply_experiment()` (missing patch imports)
  and ships as a neutral no-op baseline.
- `acceptance_rate` no longer silently fabricated as `0.55`.
- Runtime flags now applied to the correctness run too (previously bench-only,
  letting score and correctness run under different configs).
- Real-mode baseline fails loud (exit 1) when unprepared, instead of recording a
  fabricated baseline as the best score.
