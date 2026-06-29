# autoresearch-ggml-optimizer

Autonomous research project for an **on-demand pre-run GGML graph optimizer**.

The goal is to discover graph-level optimizations that run after a GGML computation graph is built and before it is scheduled/executed, improving latency, memory, or throughput without changing model outputs.

## How it works

This repo is intentionally small. Three files matter:

- **`graph.py`** — read-only GGML graph data model and builders. Do not modify.
- **`harness.py`** — read-only evaluation harness: loads graphs, applies an optimizer, measures latency/memory/correctness, and logs results. Do not modify.
- **`optimizer.py`** — the single file the agent edits. Implement optimization passes here.
- **`program.md`** — instructions for the autonomous agent.

The agent modifies `optimizer.py`, runs the harness, and keeps changes that improve the metric.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Run the baseline harness
uv run harness.py --baseline

# 3. Run with the current optimizer
uv run harness.py
```

## Metric

The primary metric is **normalized cost**: a weighted combination of:

- simulated latency (dominant for optimization)
- peak memory
- correctness penalty (large penalty if outputs diverge)

Lower is better. See `harness.py` for the exact formula.

## Graph representation

For fast experimentation, the harness uses a Python-level GGML graph model (`graph.py`) rather than requiring a full C/C++ build. It captures:

- tensor shapes, types, and memory layouts
- operator types (`MUL_MAT`, `ADD`, `RMS_NORM`, `SILU`, `FLASH_ATTN_EXT`, etc.)
- backend assignments and data movement edges

The optimizer reads this model, rewrites it, and returns a new graph. The harness simulates execution cost and checks equivalence.

## Future integration path

Once promising passes are found in Python, the plan is to port them into the actual GGML C/C++ pipeline between `ggml_build_forward_expand` and `ggml_backend_sched_alloc_graph`.
