# Changelog

All notable changes to autoggml are documented here. The project has not tagged
a release yet; everything currently lives under `[Unreleased]`.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Significance-gated scoring: keep/revert only on improvements exceeding `k·σ`
  (`--significance`, default `k=1.0`). Score uncertainty propagated via
  relative-error over the multiplicative metric (`uncertainty.py`).
- No-fabrication real mode: every metric is measured or the run raises; `--simulate`
  opt-in for plumbing/CI (never writes best-score or git state).
- `peak_mem_GiB` measured via `/usr/bin/time -v`; `acceptance_rate` parsed when
  reported, else neutral `1.0` (no more fabricated `0.55`).
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
- Keep/revert decision is now significance-gated (was raw `score > best`).
- Metric is throughput-per-memory: `score = decode·prefill·acceptance / peak_mem`.
  Dropped `build_time_s` from the score: with ccache + preserved build dir it is
  cache-state-dependent and would make runs non-reproducible. `build_time_s` is
  still measured and reported.
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
