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
The harness carries `--profile` for exactly this. Never swing before knowing the wall.

## Algorithmic — this is where we actually leapfrog

1. **Tree speculative decoding (Medusa / EAGLE-2 tree).** Verify a *tree* of
   candidate tokens per target pass, not a linear chain, via a single attention-masked
   batch. ~2× accepted tokens per target forward at moderate acceptance. **Absent from
   llama.cpp mainline** — biggest algorithmic gap. Hardest build (tree attention mask +
   tree-structured KV).
2. **Hidden-state drafting (EAGLE's core idea).** Draft from target *hidden states*,
   not tokens → acceptance jumps ~0.5–0.6 → 0.8+. Pairs with tree (= EAGLE-2/3, SOTA).
3. **Self-speculative / early-exit (no draft model).** Target's own shallow layers as
   draft, early-exit when confident. Deletes the draft model — no draft weights, no
   draft graph, no second KV. Elegant for a 27B target.
4. **Hybrid n-gram + neural draft ("lookup" speculative).** Free, near-100% acceptance
   on repetitive text and **code** (our benchmark prompts). Costs nothing on GPU. The
   blend with the neural draft inside one loop is the win.

## Architectural — exploit contention and the hardware

5. **CPU-draft + GPU-target, fully overlapped (zero GPU contention).** Tiny draft on
   CPU, target on GPU, stop fighting for the same SMs/L2. Often *the* speculative
   speedup and largely absent from naive ggml speculative.
6. **The moat: UMA-native draft on Strix Halo.** CPU and iGPU share one memory pool —
   draft lives in system memory, feeds the target with **zero copy**, KV handoff is a
   pointer not a DMA. Generic ggml treats the 8060S as a generic Vulkan device and pays
   copies it need not. A lockless, pipelined, UMA-aware speculative path is a
   hardware-specific 1.5–2× that won't show on any NVIDIA box — and NVIDIA is where
   llama.cpp's attention goes. **This is our defensible edge.**
7. **Double-buffer draft⇄verify.** Overlap draft step *N+1* with target-verify *N*.
   Halves the critical path once (5) is in place.

## Kernel / graph — cut overhead llama.cpp leaves on the table

8. **CUDA-graph-capture the verification subgraph.** Fixed shapes → capture the verify
   pass → ~zero CPU launch overhead. Big for decode latency on the 3090. Verify batch
   may not be on llama.cpp's captured path.
9. **Fuse the draft subgraph into one kernel.** Draft = pile of small ops →
   CPU-dispatch-bound → GPU starves. One fused draft kernel kills launch overhead.
   **Matters more on Vulkan/Strix Halo** (higher dispatch overhead) → bigger win exactly
   where we have a hardware edge.
10. **Branch-aware KV with O(1) rollback + shared-prefix compute.** Stop recomputing the
    K-candidate shared prefix K times; rollback by pointer (journal/append-only KV), no
    memmove. Pure kernel win, measurable immediately.
11. **Aggressive draft quantization (Q2_K/Q3_K).** Draft only needs plausible proposals
    → quantize hard → smaller footprint/bandwidth → bigger draft or higher K for same
    cost. Trivial harness sweep.
12. **Specialized Q4_K verify-batch kernel + Vulkan kernel tuning.** Verify batch is an
    odd shape (narrow batch, long shared prefix) that may miss ggml's tuned paths. On
    **Vulkan/Strix Halo Q4_K is far less tuned** — subgroup-shuffle layouts, cooperative
    matrix, packed layouts all under-explored. Real headroom on our second target.

## Runtime — cheap; the harness proves them in an afternoon

13. **Adaptive K controller.** Fixed K leaves speed on the table; optimal K is closed-form
    from rolling acceptance *p* and draft/target speed ratio. ~20-line controller.
14. **KV-cache quant + GQA sweep.** Existing knobs (`cache-type-k/v`); harness finds the
    Pareto frontier vs acceptance hit.

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
- **Meta (15, 16, 17):** harness-extension track, **not** a lucebox patch — maintainer
  work, outside the experiment agent's read-only contract on the harness / runner.
  `benchmarks/` → workload grid (16); `autoluce/parallel/runner.py` → multi-target dispatch (15); new
  `select_config` surface in `experiment.py` (17). Scored as a Pareto policy over the
  hardware × workload grid, significance-gated across the joint distribution.
