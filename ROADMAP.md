# autoggml optimization roadmap — beating llama.cpp

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

## Execution order

- **Profile both targets first** — know the wall before swinging (`--profile`).
- **Land the cheap/high-EV trio the harness validates immediately:** adaptive K (13),
  n-gram hybrid (4), CUDA-graph verify flag sweep (8). Small patches, real numbers.
- **One big algorithmic bet: tree speculative + hidden-state drafting (1+2).** This is
  the move that *beats* llama.cpp rather than trails it. Validate via the harness.
- **Pursue the moat: UMA-native draft on Strix Halo (6).** The result no NVIDIA-focused
  competitor reproduces. Turns the second box from a regression gate into lucebox's
  signature win.

## Mapping to harness experiment categories

- **Runtime (13, 14):** `experiment.get_runtime_flags()` + adaptive controller patch →
  measurable now, no kernel work.
- **Build/graph (8, 9):** CMake flags + patch helpers in `patches.py`; build-time diff
  via `build_time_s`, runtime diff via decode tok/s.
- **Kernel (10, 11, 12):** source patches against `ggml/` and `common/speculative.cpp`;
  correctness gate (perplexity, pending) must hold.
- **Algorithmic (1, 2, 3, 4):** larger patches, possibly under `patches/` as files;
  acceptance_rate and decode tok/s are the headline metrics.
- **Architectural (5, 6, 7):** cross-backend orchestration; depends on Phase 2 targets /
  serve-once work and the Strix Halo host.
