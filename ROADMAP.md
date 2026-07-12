# autoluce optimization roadmap — beating llama.cpp

Living document. The frame, the ranked ideas, and the execution order. Source of
truth for experiment selection so we don't chase noise.

## Strategic frame

Do **not** try to out-kernel llama.cpp on a generic CUDA matmul — they have more
people and it is a flat fight. lucebox wins only on two axes:

- **Structure llama.cpp's generic engine can't exploit** (speculative-decoding-specific).
- **Our specific hardware** — RTX 3090 (CUDA) + AMD Strix Halo (unified memory, Vulkan/HIP) — that generic ggml leaves on the table.

Everything below is one of those two.

## 0. Measure before you build (de-risks the whole roadmap)

One experiment decides everything else: a profiler pass — **Nsight Systems on the
3090, Radeon GPU Profiler / rocprof on Strix Halo** — to learn whether decode is
**launch-bound, memory-bandwidth-bound, or compute-bound**. The two targets will
disagree, and the correct kernel/graph work is the opposite depending on the answer.
`autoluce/bench/harness.py` `main()` exposes `--profile` for exactly this: it captures
an nsys/rocprof trace per benchmark into `results/profiles/`, forwarded over SSH
targets. `cli.py` additionally provides `profile-report` to summarize rocprofv3
captures. Never swing before knowing the wall.

### Parity contract (mandatory for any A/B number)

Any lucebox-vs-llama.cpp (or A-vs-B) number is **void** unless all six axes are
identical on both arms and logged:

1. Same GGUF file + quant — path asserted.
2. Same `cache-type-k` / `cache-type-v`.
3. Same `--max-ctx` — KV reservation changes the decode regime even at equal prompt
   length.
4. Same sampling — temp + seed, prefer temp 0.
5. Enough repetitions that **both** arms have converged past warmup.
6. Caching state symmetric and **proven** by reading `pt`/`prefix_len` on both arms —
   either both prefix-cache (system anchor) or both reprocess (distinct prompts per
   turn).

Report **median** (plus mean/best/min) from the raw per-turn arrays — never mean alone
— and always all three speeds: prefill tok/s, decode tok/s, wall. Target-only campaigns
(no B arm) are exempt from arm matching but must still log every axis in the result
bundle.

## Algorithmic — this is where we actually leapfrog

1. **Tree speculative decoding (Medusa / EAGLE-2 tree).** Verify a *tree* of
   candidate tokens per target pass, not a linear chain, via a single attention-masked
   batch. ~2× accepted tokens per target forward at moderate acceptance. **Absent from
   llama.cpp mainline** — biggest algorithmic gap. Hardest build (tree attention mask +
   tree-structured KV).
   Success: acceptance rate ≥ current harness best (no regression) and end-to-end
   decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA); exact-output gate
   green; revert = fails the harness k·sigma significance gate.
2. **Hidden-state drafting (EAGLE's core idea).** Draft from target *hidden states*,
   not tokens → acceptance jumps ~0.5–0.6 → 0.8+. Pairs with tree (= EAGLE-2/3, SOTA).
   Success: acceptance rate ≥0.8 and end-to-end decode tok/s ≥ +5% over current harness
   best on RTX 3090 (CUDA); exact-output gate green; revert = fails the harness
   k·sigma significance gate.
3. **Self-speculative / early-exit (no draft model).** Target's own shallow layers as
   draft, early-exit when confident. Deletes the draft model — no draft weights, no
   draft graph, no second KV. Elegant for a 27B target.
   Success: acceptance rate ≥ current harness best (no regression) and end-to-end
   decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA); exact-output gate
   green; revert = fails the harness k·sigma significance gate.
4. **Hybrid n-gram + neural draft ("lookup" speculative).** Free on repetitive text,
   costs nothing on GPU. `benchmarks/prompts.txt` is three prose/trivia prompts and one
   trivial factorial, not code — so no acceptance-rate claim on code is meaningful
   until a code-heavy prompt suite is added to the benchmark contract and this item is
   re-baselined against it (the same prerequisite item 16 needs). The blend with the
   neural draft inside one loop is the win.
   Success: acceptance rate ≥ current harness best (no regression) on the existing
   prompt suite (`benchmarks/prompts.txt`) and end-to-end decode tok/s ≥ +5% over
   current harness best on RTX 3090 (CUDA); exact-output gate green; revert = fails
   the harness k·sigma significance gate.

## Architectural — exploit contention and the hardware

5. **CPU-draft + GPU-target, fully overlapped (zero GPU contention).** Tiny draft on
   CPU, target on GPU, stop fighting for the same SMs/L2. Often *the* speculative
   speedup and largely absent from naive ggml speculative.
   Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
   exact-output gate green; revert = fails the harness k·sigma significance gate.
6. **The moat: UMA-native draft on Strix Halo.** CPU and iGPU share one memory pool —
   draft lives in system memory, feeds the target with **zero copy**, KV handoff is a
   pointer not a DMA. Generic ggml treats the 8060S as a generic Vulkan device and pays
   copies it need not. A lockless, pipelined, UMA-aware speculative path is a
   hardware-specific 1.5–2× that won't show on any NVIDIA box — and NVIDIA is where
   llama.cpp's attention goes. **This is our defensible edge.**
   Success: decode tok/s ≥ +5% over current harness best on Strix Halo (HIP);
   exact-output gate green; revert = fails the harness k·sigma significance gate.
7. **Double-buffer draft⇄verify.** Overlap draft step *N+1* with target-verify *N*.
   Halves the critical path once (5) is in place.
   Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
   exact-output gate green; revert = fails the harness k·sigma significance gate.

## Kernel / graph — cut overhead llama.cpp leaves on the table

8. **CUDA-graph-capture the verification subgraph.** Fixed shapes → capture the verify
   pass → ~zero CPU launch overhead. Big for decode latency on the 3090. Verify batch
   may not be on llama.cpp's captured path.
   Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
   exact-output gate green; revert = fails the harness k·sigma significance gate.
9. **Fuse the draft subgraph into one kernel.** Draft = pile of small ops →
   CPU-dispatch-bound → GPU starves. One fused draft kernel kills launch overhead.
   **Matters more on Vulkan/Strix Halo** (higher dispatch overhead) → bigger win exactly
   where we have a hardware edge.
   Success: decode tok/s ≥ +5% over current harness best on Strix Halo (HIP);
   exact-output gate green; revert = fails the harness k·sigma significance gate.
10. **Branch-aware KV with O(1) rollback + shared-prefix compute.** Stop recomputing the
    K-candidate shared prefix K times; rollback by pointer (journal/append-only KV), no
    memmove. Pure kernel win, measurable immediately.
    Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
    exact-output gate green; revert = fails the harness k·sigma significance gate.
11. **Aggressive draft quantization (Q2_K/Q3_K).** Draft only needs plausible proposals
    → quantize hard → smaller footprint/bandwidth → bigger draft or higher K for same
    cost. Trivial harness sweep.
    Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
    exact-output gate green; revert = fails the harness k·sigma significance gate.
12. **Specialized Q4_K verify-batch kernel + Vulkan kernel tuning.** Verify batch is an
    odd shape (narrow batch, long shared prefix) that may miss ggml's tuned paths. On
    **Vulkan/Strix Halo Q4_K is far less tuned** — subgroup-shuffle layouts, cooperative
    matrix, packed layouts all under-explored. Real headroom on our second target.
    Success: decode tok/s ≥ +5% over current harness best on Strix Halo (HIP);
    exact-output gate green; revert = fails the harness k·sigma significance gate.

## Runtime — cheap; the harness proves them in an afternoon

13. **Adaptive K controller.** Fixed K leaves speed on the table; optimal K is closed-form
    from rolling acceptance *p* and draft/target speed ratio. ~20-line controller.
    Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
    exact-output gate green; revert = fails the harness k·sigma significance gate.
14. **KV-cache quant + GQA sweep.** Existing knobs (`cache-type-k/v`); harness finds the
    Pareto frontier vs acceptance hit.
    Success: decode tok/s ≥ +5% over current harness best on RTX 3090 (CUDA);
    exact-output gate green; revert = fails the harness k·sigma significance gate.

## Meta — dynamic, machine- & request-aware compilation

The harness today assumes one machine, one fixed benchmark, one scalar score. That
makes a *dynamic* build — one that adapts its flags, kernels, and code paths to the
host hardware and the incoming request — structurally invisible: a per-target policy
either looks like noise (it picks the same config on the reference box) or gets kept
for a win that is a regression on a box you can't see, and the significance gate has
no way to reason about a payoff *distribution* over hardware × workload. The fix is
harness scope, not a patch to lucebox. Four enabling moves; together they turn
"compile for this machine, this request" from a static flag sweep into a discovered
policy.

**The reframe — the grid is a quality-diversity archive, not just a scoreboard.** A single
scalar frontier (`LockedFrontier`) is a hill-climber: it converges fast on the flag-sweep
basin and then sticks. The move beyond sweeps is population search — AlphaEvolve / OpenEvolve
/ Sakana's evolutionary CUDA engineer maintain an *archive* indexed by behavior descriptors
with one elite per cell, reward novelty + fitness, and let the LLM mutate *and recombine*
across cells. This domain's behavior descriptors are exactly the axes below (machine ×
workload × technique-family), so building the grid for scoring and building the archive for
novelty are the **same work**. Prior art: AutoKernel is this harness's structural twin (same
one-file / keep-revert loop, arXiv:2603.21331) — but it too is single-frontier hill-climbing;
the population/archive layer is the open frontier, and it is higher-leverage here because
speculative decoding's structure (tree verify, hidden-state drafting, UMA zero-copy) is far
less trodden than AutoKernel's matmul/softmax/attention kernels.

15. **Multi-target scoring = archive axis 1 (machine).** Run the loop on both boxes, score
    per-target, and let an experiment's payoff be a *vector* over hardware, not a scalar.
    Each machine becomes a column in the archive with its own elite, so a Strix-Halo win and
    a 3090 win both persist instead of one evicting the other. Without this, "machine
    constraints" is unobservable; today the second box is a regression gate, after this it
    is a first-class objective. Depends on the Strix Halo host reachable from the parallel runner.
16. **Workload suite = archive axis 2 (request).** Vary context length, batch shape, and
    prompt type (code / prose / long-context) so each (machine × workload) cell holds its
    own elite. Single-box useful immediately: a "win" that only helps short-context stops
    evicting the long-context elite. `benchmarks/` is fixed prompts today — the request
    dimension is absent.
17. **Config-selector experiment type.** A new experiment surface:
    `select_config(machine_features, request_features) → (cmake_flags, runtime_flags,
    code_paths)`. The harness scores the *policy* over the hardware × workload grid,
    significance-gated across the joint distribution — not a single config. This is the
    surface that lets the agent discover, e.g., "Vulkan path for Strix-Halo
    long-context, CUDA path for 3090 batch-N." Depends on 15 and 16 for a grid to
    select over.
18. **Novelty + recombination engine (the operator over the archive).** The archive
    (15 × 16) is just storage until the search uses it. Two LLM-driven operators:
    (a) novelty-weighted selection — sample an under-explored cell rather than always
    mutating the global best, so the search leaves the flag-sweep basin; (b) crossover —
    hand the agent two cell elites (e.g. adaptive-K controller + KV-quant sweep) and ask
    for a child combining both. `LockedFrontier.claim_best_if_significant` generalizes to
    `insert_if_elite_in_cell(behavior, candidate)`; a `merge_patches` helper feeds two
    `patches/*.patch` files as crossover input. This is the mechanism that discovers
    structure hill-climbing cannot.

The static subset (best flags for the reference box) the harness can already find:
`experiment.get_cmake_flags()`, `patches.apply_march_native` ("compile for THIS CPU"),
and items 6 and 12 are all per-target static tuning. 15–18 generalize that from one box
to the grid.

**Coordination layer — shipped.** `concurrency.LockedFrontier` (file-locked frontier +
atomic `claim_best_if_significant`, re-verifying against the live best), `worktree`
helpers, and `runner.run_parallel` (live-frontier funnel) are in. `agent_loop` now routes
every shared-state write through the lock and honors `AUTOLUCE_FRONTIER`, so N workers in
N worktrees -- or N hosts into one shared checkout -- converge safely instead of racing on
`.best_score.json` / `results.tsv` / the build dir. This is the foundation #15 and #17
build on; it does **not** yet make the score a per-target vector, which is the remaining
#15 work.

**Move 1 — shipped.** `selector.rank_by_bottleneck` + `autoluce ideas --bound
<memory|compute|overhead>` rank untried items so those targeting the active profiling
bottleneck come first (the AutoKernel Amdahl-targeting move). Pure decision, tested; the
nsys/rocprof trace-parser that auto-produces the bound verdict is the deferred I/O seam.

## NVFP4 × SM86 co-design — we own the converter (added 2026-07-11)

Context: upstream's NVFP4 MMVQ fusion (llama.cpp `3899b39ce`) folds only the
post-scale + bias into the epilogue — 1.02–1.09×, every number measured on
Blackwell-class hardware. On the 3090 (no FP4 tensor cores) batch-1 decode is
DRAM-bound: ~936 GB/s over ~14.4 GB of touched weights (27B @ ~4.25 bpw) caps
autoregressive decode near **~65 tok/s**, so epilogue fusion is noise there — and the
frontier decode targets (>66–82 tok/s) sit *above* that ceiling, reachable only by
amortizing each weight fetch across speculation. Copying the upstream fusion is not
the move; changing *what gets fused over* is.

19. **Fixed-width NVFP4 verify-batch kernel (ncols = n_draft+1 = 16) + converter
    co-design.** dflash's verify pass is a fixed 16-wide matvec (golden pins
    `n_draft: 15`) — no-man's land between MMVQ's specialized ncols 1–8 and MMQ's
    ≥32 tiles, and llama.cpp can't hardcode it without breaking generality. One
    weight fetch amortized across all 16 candidates multiplies the bandwidth ceiling
    per accepted token. Two SM86 tricks stack: (a) E2M1 values doubled are exact
    small integers {0..12} → lossless FP4→INT8 map, fold the ×½ into the FP8 block
    scale, ride dp4a; (b) `autoluce nvfp4 convert` owns the GGUF → repack offline
    into an SM86-native layout (scales segregated contiguous for vectorized loads,
    nibbles pre-swizzled to dp4a lane order, 128-bit aligned, cp.async
    double-buffered). Upstream must keep canonical layout for every backend; we
    co-design file format and kernel. `native/nvfp4/` (W4A16 GEMV + CPU oracle) is
    the starting lab.
    Success: end-to-end decode tok/s ≥ +10% over current harness best at 16K on
    RTX 3090 (CUDA); exact-output gate green; revert = fails the harness k·sigma
    significance gate.

20. **Speculation along the depth axis: draft-guided sparse prefill.** The prefill
    analog of 19. Prefill is compute-bound at 1K but the measured 1,110 → 368 tok/s
    fall to 128K is attention's O(n²) — at depth, prefill is attention/KV-traffic
    bound, and the scarce resource flips from weight bytes to KV reads. Decode's
    trick amortizes weight reads across *future* tokens (speculation width); the
    prefill twin lets the cheap draft pay the O(n²) toll over *past* tokens: the
    draft must prefill the same prompt anyway to draft at all, so its attention
    pattern is a free oracle for which KV blocks matter — the target then runs
    block-sparse prefill attention over the selected blocks, dense on the rest.
    llama.cpp has no draft-model concept inside prefill, so this structure is
    invisible to it. Catch, stated honestly: this is **not output-exact** (logits
    shift), so the exact gate fails by design — it needs the KL oracle
    (`autoluce/bench/kl.py`), which fails closed until Lucebox exposes a logits
    endpoint. Park behind that dependency; do not run it against the exact gate.
    Success: prefill tok/s ≥ +15% over current harness best at 64K/128K on RTX 3090
    (CUDA) with KL divergence under the frozen-baseline gate; revert = fails either
    gate.

21. **Depth-adaptive prefill chunk schedule (the lossless sibling of 20).** The
    campaign already proved per-depth winners (557.9 → 637.5 @64K, 368.0 → 407.7
    @128K via ubatch/qbatch 3072, KV tile 8192) — but the winning shape differs per
    cell, and llama.cpp fixes ubatch for an entire prefill pass. Make the chunk/tile
    shape a function of *current* depth within one pass: as the attention share grows
    with each chunk, the optimal ubatch/KV-tile moves, and a `ubatch(depth)` schedule
    captures every cell's winner in a single prefill. Exact output preserved;
    measurable today with the existing campaign cells and env controls
    (`DFLASH_PREFILL_UBATCH`, `DFLASH_CHUNKED_Q_BATCH`, `DFLASH_CHUNKED_CHUNK`).
    Success: prefill tok/s ≥ +5% over the best measured cell (637.5 @64K, 407.7
    @128K) on RTX 3090 (CUDA); exact-output gate green; revert = fails the harness
    k·sigma significance gate.

## Execution order

- **Profile both targets first** — know the wall before swinging (`--profile`).
- **Land the cheap/high-EV trio the harness validates immediately:** adaptive K (13),
  n-gram hybrid (4), CUDA-graph verify flag sweep (8). Small patches, real numbers.
- **One big algorithmic bet: tree speculative + hidden-state drafting (1+2).** This is
  the move that *beats* llama.cpp rather than trails it. Validate via the harness.
- **Stand up the grid before the moat (do-order 16 → 15 → 18 → 17).** The Strix Halo moat
  (6) and the Vulkan tuning (12) are per-target wins — but the harness scores one box,
  so a per-target gain is invisible or mis-attributed. Build the workload axis (16) first on
  the reference box (cheapest, unblocks signal on every later item), then the machine axis
  (15), then the novelty/recombination operator (18) that turns the grid into a
  quality-diversity archive, then the config-selector surface (17) as the capstone that
  reads the archive. Numbered 15–18 but executed 16 → 15 → 18 → 17.
- **Pursue the moat: UMA-native draft on Strix Halo (6).** The result no NVIDIA-focused
  competitor reproduces. Turns the second box from a regression gate into lucebox's
  signature win.

## Mapping to harness experiment categories

- **Runtime (13, 14):** `experiment.get_runtime_flags()` + adaptive controller patch →
  measurable now, no kernel work.
- **Build/graph (8, 9):** CMake flags + patch helpers in `autoluce/loop/patches.py`; build-time diff
  via `build_time_s`, runtime diff via decode tok/s.
- **Kernel (10, 11, 12):** source patches against `ggml/` and `common/speculative.cpp`;
  correctness gate (golden outputs + KL oracle, `autoluce/bench/kl.py`) must hold.
- **Algorithmic (1, 2, 3, 4):** larger patches, possibly under `patches/` as files;
  decode tok/s is the headline metric (acceptance_rate is a logged diagnostic, not scored).
- **Architectural (5, 6, 7):** cross-backend orchestration; depends on Phase 2 targets /
  serve-once work and the Strix Halo host.
- **NVFP4 × SM86 co-design (19, 20, 21):** 19 and 21 are kernel/runtime work against
  the exact gate (`native/nvfp4/` lab, prefill campaign cells); 20 is algorithmic and
  **blocked on the Lucebox logits endpoint** for the KL gate — do not run it exact-gated.
- **Meta (15, 16, 17):** harness-extension track, **not** a lucebox patch — maintainer
  work, outside the experiment agent's read-only contract on the harness / runner.
  `benchmarks/` → workload grid (16); `autoluce/parallel/runner.py` → multi-target dispatch (15); new
  `select_config` surface in `experiment.py` (17). Scored as a Pareto policy over the
  hardware × workload grid, significance-gated across the joint distribution.
