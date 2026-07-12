# Changelog

All notable changes to AutoLuce are documented here. The project has not tagged
a release yet; everything currently lives under `[Unreleased]`.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Runtime shared-library provenance**: real product runs now resolve the executable's
  dynamic dependency closure under the effective server environment and record canonical
  paths, sizes, and SHA-256 hashes. Mutation of `libggml-cuda`, GGML CPU/base libraries,
  CUDA libraries, or any other loaded dependency invalidates the run even when the
  `dflash_server` bytes did not change. Baseline metrics are now saved only after this
  final stability gate passes.
- **Lazy Qwen3.6 rollback-cache candidate**: added a product patch that leaves
  target-only prefill genuinely prefill-only and promotes the cache only after a request
  qualifies for speculative decoding. An isolated exact-output 8K trace removed exactly
  1,392 first-request synchronous tensor uploads and avoids about 1.31 GiB of unused
  rollback storage. Plain chain speculation is included in the promotion predicate.
  Promotion remains blocked on speculative E2E because the current central CUDA
  dispatcher fails first with an unrelated invalid-pitch error.
- **Tensor-core vendor-sync decomposition**: isolated the smallest buildable CUDA
  tensor-core closure from the rejected 61-file vendor patch and reran a fully
  content-addressed clean/subset/subset/clean comparison. It reproduced +4.55% IQ4_XS
  prefill at 1K and +4.58% at 8K, but also reproduced deterministic greedy-token
  divergence. The closure is rejected; the result localizes both effects to
  FAttn/MMQ/quantization and their shared interfaces, narrowing the next split to
  surgical interface hunks rather than broad file families.
- **MMQ stream-k causal isolation**: extracted a standalone one-file scheduler/fixup
  patch and ran a content-addressed candidate/clean/candidate/clean comparison. It
  improved IQ4_XS prefill by 6.02% at 1K and 5.71% at 8K in both directions, while
  exactly reproducing the rejected tensor closure's candidate-specific output sequence.
  This localizes both effects to MMQ reduction scheduling and rules stale FAttn out as
  the leading cause for this gap. The quality promotion gate now passes: five targeted
  IQ4_XS CUDA-vs-CPU operator cases, 34 matched full-vocabulary logit samples, bit-exact
  repeat captures, and 20 longer generation canaries. Combined mean KL is 0.002729,
  maximum KL is 0.016672, and top-20 overlap never falls below 90%. The patch is an
  exact backport of upstream llama.cpp #22298 and is ready for a Lucebox Hub PR.
- **Product first-token logits diagnostics**: added a fail-closed AutoLuce parser and
  numerical quality oracle plus a Lucebox candidate patch for opt-in, non-streaming raw
  F32 logits at the final prompt position. Diagnostic requests bypass prefix/disk-cache
  restore, validate every value as finite, and leave ordinary requests unchanged. The
  product build passes all 2,041 server-unit assertions; AutoLuce gates mean/maximum KL,
  top-k overlap, absolute error, margins, and argmax movement.
- **Content-addressed vendor-sync ABBA**: reran the 61-file CUDA vendor patch with one
  immutable GDN-enabled executable and separately hashed clean/vendor GGML CUDA
  libraries. The vendor arm reproducibly improved IQ4_XS prefill by 4.24% at 1K and
  4.60% at 8K, clearing the raw-standard-deviation gate in both ABBA directions, but
  deterministically changed exact greedy tokens. The broad patch is rejected pending
  correctness-gated decomposition; the result confirms stale vendor performance without
  over-attributing it to FlashAttention alone.
- **Content-addressed product/vendor evidence**: every real run now records the exact
  Lucebox revision, working product-tree digest, independent vendored-GGML digest,
  runtime binary SHA-256, dirty paths, selected backend, and separate product versus
  vendor backend contracts. The harness snapshots this identity immediately after the
  build and rejects measurements if source or binary bytes change before completion,
  preventing shared-checkout and stale-binary results from entering the frontier. A
  nonblocking per-checkout lease also stops cooperating local agents before concurrent
  reset/build/measure cycles can contaminate each other.
- **Qwen3.6 GDN broadcast forensic candidate**: retained a product patch that removes
  redundant 16-to-48-head Q/K repeats already supported by the fused CUDA GDN kernel,
  with exact-output ABBA evidence. A stable 14+14 repetition comparison improved
  IQ4_XS prefill by 0.89% at 1K and 1.17% at 8K. The 8K result clears the conservative
  raw-standard-deviation gate and is retained; the 1K result remains inconclusive.
- **RTX 3090 prefill gap forensics**: corrected the original 9.1% 8K Luce/upstream
  estimate with same-session controls to 3.36%, then reduced it to roughly 2.2% with
  the retained GDN broadcast patch. CUDA API backtraces identified a target-only
  first-request rollback migration that allocates about 1.31 GiB and performs 1,392
  synchronous uploads. Steady-state traces ruled synchronization count out as the
  residual throughput cause. GDN in-place state, upstream-style `set_rows`, target-only
  feature capture, and F16 FWHT removal are recorded as rejected or inconclusive.
- **F16 KV rotation negative result**: retained and rejected a candidate that bypasses
  FWHT for unquantized F16/BF16 K caches. It changed exact greedy output, slowed the 1K
  median, and improved the noisy 8K median by only 0.9%, ruling it out before a reverse
  run and narrowing the parity investigation toward KV/attention layout and graph life.
- **Native Lucebox environment controls**: inherited `DFLASH*`, `GGML_*`, and `LUCE_*`
  settings are now passed through to managed servers and recorded as effective
  experiment provenance. Explicit `runtime_env` values retain precedence and can
  unset inherited controls. Stable `DFLASH_PREFILL_UBATCH`, `DFLASH_CHUNKED_Q_BATCH`,
  and `DFLASH_CHUNKED_CHUNK` aliases map to Lucebox's current `DFLASH27B_*` names.
- **Context-validated RTX 3090 NVFP4 prefill campaign**: a target-only Qwen3.6-27B
  benchmark now generates deterministic prompts for 1K/16K/64K/128K cells, rejects
  mislabeled depths using authoritative server token counts, scores `prefill_tok_s`
  from the benchmark objective, reserves output headroom separately, and freezes a
  local exact reference through `autoluce freeze --benchmark`. Diagnostic context and
  repetition overrides are recorded in result bundles. Product-native uppercase
  `DFLASH*` and `GGML_*` variables are passed only to the managed server and retained
  as experiment evidence.
- **Correct vendored GGML patch application**: vendor patches now run from the Lucebox
  Git worktree with `--directory=server/deps/llama.cpp` and a reverse applicability
  check. This fixes the non-submodule vendor layout and prevents a no-op patch from
  being reported as applied.
- **Lucebox HTTP benchmark and exact-quality adapter**: the live harness launches the
  pinned product's `dflash_server`, disables prefix/prefill caches for measurement,
  consumes authoritative `usage.timings` prefill/decode fields, records acceptance and
  process/UMA telemetry, and compares deterministic completions with frozen goldens.
  Golden generation and remote `freeze` use the same client and server lifecycle, so
  measurement and quality cannot drift onto a standalone llama.cpp binary. Product KL
  requirements fail closed because the current API does not expose token logits. The
  adapter was also exercised end to end with the pinned product, a local Qwen3.6 GGUF,
  and the RTX 3090 CUDA backend.
- **RTX 3090 NVFP4 CUDA laboratory**: `autoluce nvfp4 test|bench` builds a bounded SM86
  target with CUDA 12.6 and at most four jobs. It includes packed E2M1 weights, E4M3
  block scales, a global FP32 scale, fused W4A16 GEMV, an independent CPU oracle, and a
  naive FP16 control. On the local RTX 3090, the validated 4096x4096 operator measured
  0.0419 ms versus 0.0579 ms for the control (1.38x, 200 iterations, zero max error).
- **Manifest-driven Lucebox ownership**: `sources/lucebox.toml` pins the complete
  `Luce-Org/lucebox-hub` product and declares its checkout layout, product CMake root,
  supported CUDA/HIP backends, build targets, runtime, capabilities, and expected GGML
  vendor provenance. `autoluce source status` reports the contract and `autoluce source
  check --remote` detects upstream movement; scheduled CI runs the drift check weekly.
- **Vendored-source guardrails**: setup, reset, patch application, agent worktrees, and
  agent patch allowlists now share `SourceLayout`. Product patches are Hub-relative;
  vendor patches use the explicit `AUTOLUCE_PATCH_SCOPE=vendor` boundary. Setup validates
  `server/deps/llama.cpp/VENDOR.md` before building the real `dflash_server`,
  `test_dflash`, and `test_deepseek4_unit` product targets, and initializes the declared
  Block-Sparse Attention submodule so CUDA does not silently lose BSA.
- **Cooperative agent challenges** (`autoluce agent ...`): agents register `implement`,
  `review`, or `recombine` capabilities; choose bounded task packets; claim expiring
  leases; and work from a challenge-pinned commit in isolated Git worktrees. Parallel
  implementations remain blind until measured, then feed a review and credited
  recombination stage. `agent next|start|submit|advance|status|card` support structured
  JSON for agent callers, while a remembered `0600` identity keeps the normal CLI short.
- **Agent evidence and safety model**: task packets include the objective, distinct
  approach, profiler evidence, expected impact, difficulty, token/time budgets, allowed
  and forbidden paths, definition of done, and validation command. A single patch gate
  protects benchmarks, goldens, contracts, models, and verifier code. Challenge cards
  preserve contributor/source graphs, measured rankings, execution failures, negative
  results, and explicit `inconclusive` outcomes.
- **Distributed team coordinator**: `join`, `submit`, `status`, `pause`, `resume`,
  `leave`, `worker`, and `coordinator` provide an authenticated typed HTTP/file-backed
  queue. Candidate patches are content-addressed, each physical machine receives at
  most one active experiment, and restricted workers run only the existing
  correctness-gated pipeline under the accelerator lease. CUDA and HIP retain
  separate build directories with compilation capped at four jobs.
- **Machine-aware DeepSeek V4 / Strix Halo workflow**: `doctor`, `consult`, `freeze`,
  `test-drive`, `onboard`, `verify`, and `profile-report`; stable machine/model
  fingerprints; versioned research contracts; external and sharded GGUF catalog entries;
  HIP remote execution; UMA memory, fault, swap, GTT/VRAM, temperature, power, and
  clock telemetry; context-conditioned 8K/32K/128K cells; and machine-scoped evidence.
  Inventory, contracts, onboarding, HTTP measurement, and exact quality capture are
  active. Product KL capture remains gated on a Lucebox token-logits endpoint.
- **DeepSeek V4 fused Sinkhorn reference**: checksummed upstream patch retained as
  research evidence. It is marked `requires-port-and-reprofile` because it targets the
  former standalone tree, while current Hub code owns DeepSeek V4 separately and already
  contains custom fused HC CUDA/HIP device paths.
- **KL quality oracle** (`kl.py`, `autoluce kl-base <benchmark>`): the original
  standalone adapter checks candidates against frozen reference logits and rejects mean
  or maximum KL violations. Its parser and gates remain tested, but product execution is
  fail-closed until equivalent capture exists through Lucebox Hub.
- **Shadow bench** (`shadow.py`, `autoluce shadow proxy|build`): a local capture proxy
  turns private traffic into a deduplicated workload. Capture and benchmark construction
  remain available; KL scoring awaits a product token-logits endpoint. Prompts stay in
  gitignored local storage.
- Unified `autoluce` CLI (`cli.py`): `uv run autoluce <command>` routes to focused
  package modules without reimplementing their behavior; `autoluce help` is the current
  command inventory. One-liner `install.sh` (`curl | bash`) clones, ensures `uv`, syncs,
  and prints next steps.
- Product GPU auto-detection: setup selects CUDA or HIP and fails before CMake when the
  detected backend is outside the Hub product contract. Existing GGUFs are reused from
  the HF cache, LM Studio, `~/models`, and `AUTOLUCE_MODELS` before downloading.
- Parallel-safe shared leaderboard (`concurrency.LockedFrontier`): file-locked `.best_score.json` + `results.tsv`; `claim_best_if_significant` re-verifies against the **live** frontier under the lock, so concurrent workers can't keep a stale-snapshot win. `worktree.py` isolates per-worker trees; `runner.run_parallel` fans out and funnels through the frontier; `agent_loop` honors `AUTOLUCE_FRONTIER` so worktree workers share one leaderboard.
- Profile-driven ideation (`selector.rank_by_bottleneck`): `ideas --bound <memory|compute|overhead>` ranks untried `ROADMAP.md` items so those targeting the active bottleneck come first.
- Optional embedded LLM (`llm.py` + `propose.py`): one OpenAI-compatible client for cloud or local (`llama-server`/Ollama/vLLM/LM Studio), env-gated by `OPENAI_BASE_URL` (disabled by default; never fires silently). `propose` asks the model for the next experiment given the ranked ideas + current best.
- `ROADMAP.md` "Meta" section (#15–18): the machine × workload grid reframed as a quality-diversity archive (one elite per cell) with novelty + recombination (#18) as the operator that moves beyond hill-climbing — grounded in AutoKernel (`arXiv:2603.21331`) / AlphaEvolve prior art.
- Significance-gated scoring: keep/revert only on improvements exceeding `k·σ`
  (`--significance`, default `k=1.0`). Score uncertainty propagated via
  relative-error over the multiplicative metric (`uncertainty.py`).
- No-fabrication real mode: every metric is measured or the run raises; `--simulate`
  opt-in for plumbing/CI (never writes best-score or git state).
- `peak_mem_GiB` combines live server RSS with available VRAM/GTT telemetry for the
  product HTTP runtime; legacy command helpers retain `/usr/bin/time -v` parsing.
- Exact correctness generation and measurement share the same `DflashHttpClient` and
  deterministic request construction; empty/placeholder reference sets fail closed.
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
- Renamed the project and public namespace to **AutoLuce**: the console command is
  `autoluce`, the Python package is `autoluce`, environment variables use `AUTOLUCE_*`,
  and user configuration/state lives under `~/.config/autoluce`,
  `~/.local/share/autoluce`, and `~/.autoluce`. This is an intentional breaking rename
  with no legacy command or package alias. Existing contributors should run `uv sync`
  and rejoin or move their local configuration into the new paths.
- Replaced the 500-line cumulative README with a task-oriented guide organized around
  first use, current capability status, team participation, agent participation, remote
  Lucebox onboarding, NVFP4 CUDA work, source maintenance, and development.
- Lucebox Hub's July 2026 vendorization is now the source-of-truth boundary. The old
  standalone `lucebox-ggml` checkout, root CMake flags, and `llama-bench`/`llama-cli`/
  `llama-perplexity` execution paths are not treated as product capabilities. Live
  benchmark and exact quality now use `dflash_server`; KL remains unavailable rather
  than falling back to a different engine. Simulation remains available for CI and
  coordination tests.
- Product backend selection now uses `DFLASH27B_GPU_BACKEND=cuda|hip`. Vulkan remains a
  useful generic GGML research direction, but it is not advertised as a current
  Lucebox Hub product backend.
- Fleet and agent repositories now share one process-safe atomic JSON persistence
  primitive and one stable content-ID helper rather than maintaining parallel locking
  and hashing implementations.
- Identical candidate content now reuses completed or failed hardware evidence as well
  as active jobs, avoiding duplicate accelerator work and deterministic job-ID clashes
  when competing agents converge on the same patch.
- **Repository organized as the `autoluce/` package.** Root holds `cli.py` and the
  agent-editable `experiment.py`; the engine is grouped into `bench`, `loop`,
  `ideation`, and `parallel` modules plus the focused coordination and setup modules.
  The CLI dispatches `python -m autoluce.<module>`.
- **Constrained objective replaces the product score**: `score = decode_tok_s` only; `peak_mem_GiB` / `prefill_tok_s` are now constraints declared per benchmark in an `"objective"` block (`objective.check_constraints`, k·σ significance margin; `min_frac_of_baseline` compares against `work/baseline_metrics.json` persisted by `autoluce baseline`). A violation zeroes the score exactly like a correctness failure. Speculative runs that don't report `acceptance_rate` now **raise** (the neutral-1.0 fallback is gone); acceptance stays as a logged diagnostic only. **Existing `.best_score.json` / baselines are invalid — re-measure.**
- `pyproject.toml` gains `[build-system]` (setuptools) + `py-modules`, so `uv` installs the project and the `autoluce` console script is generated (previously `[project.scripts]` was silently ignored — no `[build-system]` meant a virtual project).
- `agent_loop.py` routes all four outcomes (baseline/keep/discard/crash) through `LockedFrontier`; the old direct file writes (`load_best*`/`save_best_score`/`log_result`) are removed.
- `profiling.ROADMAP_FOR_BOUND` and `ideas.descriptions_from_results` are now public (shared by `selector`/`propose`).
- Keep/revert decision is now significance-gated (was raw `score > best`).
- `build_time_s` is measured and reported but never scored: with ccache + a
  preserved build dir it is cache-state-dependent and would make runs
  non-reproducible.
- `reset_lucebox` preserves `work/lucebox/build-*` so
  the loop gets incremental rebuilds instead of a full rebuild every run.
- `.best_score.json` and `results.tsv` persist `score_stddev` (schema widened).
- Golden generator uses the harness's shared command builder + text extraction.

### Fixed
- `agent start` now creates its isolated worktree from the pinned
  `work/lucebox` product checkout rather than the autoluce control repository, so
  the approved engine paths in submitted patches correspond to the source agents edit.
- `experiment.py` no longer crashes on `apply_experiment()` (missing patch imports)
  and ships as a neutral no-op baseline.
- `acceptance_rate` no longer silently fabricated as `0.55`.
- Runtime flags now applied to the correctness run too (previously bench-only,
  letting score and correctness run under different configs).
- Real-mode baseline fails loud (exit 1) when unprepared, instead of recording a
  fabricated baseline as the best score.

### Security
- Agent and fleet HTTP endpoints accept typed operations and patch bytes only, never
  coordinator-provided arbitrary shell commands. Agent patch paths are allowlisted and
  verifier-owned inputs are protected. Remote agent identities currently share the
  team's coordinator credential, so attribution and leases assume a trusted team;
  per-agent scoped credentials remain future hardening.
