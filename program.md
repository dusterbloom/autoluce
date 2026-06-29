# autoresearch-ggml-optimizer

This is an autonomous research project to build an **on-demand pre-run GGML graph optimizer**.

## What we are optimizing

GGML models build a computation graph (`ggml_cgraph`) for every forward pass. Today, graph-level optimizations in `llama.cpp` are mostly hand-written per architecture (fused attention, fused FFN, etc.) and applied at graph-build time. We want a **general, modular optimizer** that runs after the graph is built and before it is scheduled, rewriting the graph to reduce simulated latency and memory while preserving correctness.

## Setup

1. **Agree on a run tag** with the user, e.g., `jun29`.
2. **Create a branch**: `git checkout -b autoresearch-ggml-optimizer/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — project overview.
   - `RESEARCH.md` — prior art and design questions.
   - `graph.py` — graph data model. Do not modify.
   - `harness.py` — evaluation harness. Do not modify.
   - `optimizer.py` — baseline optimizer (identity pass). This is what you edit.
4. **Confirm the harness runs**: `uv run harness.py --baseline` should print a baseline cost.
5. **Initialize `results.tsv`** with just the header row.

## Experimentation loop

Each experiment runs the harness on one or more benchmark graphs.

**What you CAN do:**
- Modify `optimizer.py` — add, remove, or reorder optimization passes.
- Change pass parameters and heuristics.
- Add helper functions inside `optimizer.py`.

**What you CANNOT do:**
- Modify `graph.py` or `harness.py`. They are read-only.
- Install new packages or add dependencies beyond `pyproject.toml`.
- Change the evaluation metric or correctness check.

**The goal is simple: get the lowest normalized cost.** The harness prints:

```
---
cost:              1.0000
latency_ms:        12.34
peak_mem_mb:       456.7
correctness:       pass
nodes:             120
edges:             180
```

`cost` is the metric to minimize.

**VRAM** is a soft constraint; memory reductions are good, but do not explode peak memory.

**Simplicity criterion**: All else equal, simpler is better. A tiny improvement that adds 200 lines of fragile code is not worth it. A tiny improvement from deleting code is great.

## Output format

After each run, log the result to `results.tsv` (tab-separated, 5 columns):

```
commit	cost	latency_ms	peak_mem_mb	status	description
```

- `commit`: short git hash (7 chars)
- `cost`: harness cost (0.0000 for crashes)
- `latency_ms`: harness latency (0.0 for crashes)
- `peak_mem_mb`: harness memory (0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what this experiment tried

## Experiment loop

The experiment runs on a dedicated branch.

LOOP FOREVER:

1. Look at the git state.
2. Modify `optimizer.py` with one experimental idea.
3. `git commit`.
4. Run the harness: `uv run harness.py > run.log 2>&1`.
5. Read results: `grep "^cost:\|^latency_ms:\|^peak_mem_mb:\|^correctness:" run.log`.
6. If output is empty, the run crashed. Read `tail -n 50 run.log`, attempt a fix.
7. Log to `results.tsv`.
8. If `cost` improved (lower), keep the commit.
9. If `cost` is equal or worse, `git reset` back to the previous best.

**Timeout**: If the harness takes more than 2 minutes, kill it and treat as a failure.

**Crashes**: If a bug is trivial (typo, wrong attribute), fix and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the loop begins, continue autonomously until manually interrupted. If you run out of ideas, re-read `RESEARCH.md`, combine previous near-misses, or try more radical passes.

## Suggested optimization passes to explore

These are starting ideas based on `RESEARCH.md`. Do not implement all at once; try one at a time.

1. **Dead-code elimination**: remove nodes that do not contribute to outputs.
2. **Constant folding**: evaluate constant sub-graphs at optimization time.
3. **Common subexpression elimination**: merge identical nodes.
4. **Operator fusion**:
   - `RMS_NORM` + `MUL_MAT(QKV)` → fused QKV norm
   - `SILU` + `MUL` → SwiGLU fusion
   - `MUL_MAT(Q,K^T)` + causal mask + softmax + `MUL_MAT(V)` → flash attention pattern
5. **Layout optimization**: choose tensor layouts (e.g., transposed weights) that reduce dispatch overhead.
6. **Backend placement**: move nodes to the backend with the lowest simulated cost.
7. **Memory planning**: reuse intermediate buffers using lifetime analysis.
8. **Quantized intermediates**: cast F32 intermediates to F16/BF16 where allowed.
9. **Batching**: merge equivalent single-token graphs into a batched graph.
10. **Attention-specific rewrites**: replace generic attention sub-graphs with `FLASH_ATTN_EXT` when shapes permit.

Start with the simplest passes (DCE, CSE) and measure. Then build toward fusion.

## Important correctness constraints

- Optimizations must preserve floating-point equivalence within tolerance.
- Fusing ops may change numerics; the harness checks outputs against the original graph.
- Do not change the semantics of view/reshape/transpose chains unless you can prove equivalence.
- Be careful with quantization: some rewrites are only valid for FP16/FP32 tensors.

## Tips

- Keep passes small and composable. Each pass should be a pure function `Graph -> Graph`.
- Use `graph.py` helpers (`topological_sort`, `replace_uses`, `is_fusible_attention`) rather than manipulating raw edges.
- Measure after every change. It is easy to add complexity that does not improve cost.
- If a pass only helps on one benchmark, consider making it conditional or abandoning it.
