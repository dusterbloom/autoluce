"""
Evaluation harness (read-only for the agent).

Measures the quality of an optimizer by:
1. Loading one or more benchmark graphs.
2. Running the optimizer in optimizer.py on each graph.
3. Simulating latency and memory of the optimized graph.
4. Checking numerical equivalence against the original graph.
5. Computing a single cost metric.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Dict, List, Tuple

from graph import (
    DEFAULT_BACKEND_COST,
    GGMLType,
    Graph,
    OpType,
    Tensor,
    TensorId,
    build_attention_graph,
    build_simple_mlp_graph,
    dtype_bytes,
)
from optimizer import optimize


# ---------------------------------------------------------------------------
# Simulated execution cost
# ---------------------------------------------------------------------------

# Base cost per op in arbitrary units, scaled by tensor size.
BASE_OP_COST: Dict[str, float] = {
    OpType.NONE: 0.0,
    OpType.VIEW: 0.0,
    OpType.RESHAPE: 0.01,
    OpType.TRANSPOSE: 0.02,
    OpType.PERMUTE: 0.02,
    OpType.CPY: 0.5,
    OpType.DUP: 0.3,
    OpType.ADD: 0.1,
    OpType.MUL: 0.1,
    OpType.SUB: 0.1,
    OpType.DIV: 0.1,
    OpType.SILU: 0.2,
    OpType.GELU: 0.25,
    OpType.RELU: 0.05,
    OpType.SIGMOID: 0.15,
    OpType.SOFTMAX: 0.4,
    OpType.NORM: 0.3,
    OpType.RMS_NORM: 0.25,
    OpType.GROUP_NORM: 0.3,
    OpType.MUL_MAT: 1.0,
    OpType.MUL_MAT_ID: 1.1,
    OpType.SCALE: 0.05,
    OpType.GET_ROWS: 0.3,
    OpType.ROPE: 0.2,
    OpType.FLASH_ATTN_EXT: 0.6,
    OpType.FLASH_ATTN_BACK: 1.2,
    OpType.CONV_1D: 0.8,
    OpType.CONV_2D: 1.2,
    OpType.POOL_1D: 0.3,
    OpType.POOL_2D: 0.4,
    OpType.SSM_CONV: 0.5,
    OpType.SSM_SCAN: 0.5,
}

# Fused op names get a discount because they avoid round-trips.
FUSION_DISCOUNT = 0.85


def op_cost(t: Tensor) -> float:
    """Simulated latency cost of a single node (ms-ish arbitrary units)."""
    base = BASE_OP_COST.get(t.op, 0.5)
    backend_mult = DEFAULT_BACKEND_COST.get(t.backend, 1.0)
    elements = max(1, math.prod(t.shape))
    # Quantized types get a small penalty for dequant overhead.
    quant_penalty = 1.2 if t.dtype.startswith("Q") else 1.0
    if t.op in (OpType.MUL_MAT, OpType.MUL_MAT_ID):
        # Approximate GEMM cost: O(M*N*K) where shape semantics are simplified.
        cost = base * math.log1p(elements) * quant_penalty * backend_mult
    else:
        cost = base * math.log1p(elements) * quant_penalty * backend_mult
    # Fusion bonus: if the op name suggests it is fused, discount it.
    if "fused" in t.op.lower() or "flash" in t.op.lower():
        cost *= FUSION_DISCOUNT
    return cost


def simulate_latency(g: Graph) -> float:
    """Sum of simulated node costs in topological order."""
    total = 0.0
    for tid in g.topological_sort():
        total += op_cost(g.nodes[tid])
    return total


# ---------------------------------------------------------------------------
# Simulated memory planning
# ---------------------------------------------------------------------------

def simulate_peak_memory(g: Graph) -> float:
    """
    Simulate a simple reference-counting allocator.
    Returns peak live bytes (parameters + intermediate buffers).
    """
    order = g.topological_sort()
    uses = g.uses
    live: Dict[TensorId, Tensor] = {}
    peak = 0
    current = 0

    # Parameters and inputs start live.
    for tid in g.inputs:
        live[tid] = g.nodes[tid]
        current += g.nodes[tid].buf_size
    for tid, t in g.nodes.items():
        if t.is_param and tid not in live:
            live[tid] = t
            current += t.buf_size

    for tid in order:
        t = g.nodes[tid]
        if tid not in live:
            live[tid] = t
            current += t.buf_size
        peak = max(peak, current)
        # Decrement ref counts by counting remaining users.
        remaining_users = sum(1 for u in order if tid in g.nodes[u].src and u != tid)
        # Simple approximation: free if no later node uses it.
        if remaining_users == 0 and not t.is_param and tid not in g.outputs:
            current -= t.buf_size
            del live[tid]

    return peak / (1024.0 * 1024.0)  # MB


# ---------------------------------------------------------------------------
# Correctness check (simulated)
# ---------------------------------------------------------------------------

def simulate_value(t: Tensor, values: Dict[TensorId, List[float]]) -> List[float]:
    """
    Very lightweight forward evaluation producing deterministic pseudo-values.
    Not meant to be numerically accurate; meant to catch semantic changes.
    """
    if t.id in values:
        return values[t.id]
    rng = random.Random(hash(t.id.name) % (2**31))
    n = max(1, math.prod(t.shape))
    if t.op == OpType.NONE:
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op == OpType.MUL_MAT:
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV):
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op == OpType.SILU:
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op == OpType.SOFTMAX:
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op in (OpType.RMS_NORM, OpType.NORM):
        out = [rng.random() for _ in range(min(n, 256))]
    elif t.op in (OpType.RESHAPE, OpType.TRANSPOSE, OpType.PERMUTE, OpType.VIEW):
        # For shape-only ops, re-use first source deterministically.
        if t.src:
            out = values.get(t.src[0], [rng.random()])
        else:
            out = [rng.random()]
    else:
        out = [rng.random() for _ in range(min(n, 256))]
    values[t.id] = out
    return out


def check_correctness(original: Graph, optimized: Graph) -> Tuple[bool, float]:
    """
    Compare a deterministic pseudo-forward pass on both graphs.
    Returns (pass, max_relative_error).
    """
    orig_vals: Dict[TensorId, List[float]] = {}
    opt_vals: Dict[TensorId, List[float]] = {}

    for tid in original.topological_sort():
        simulate_value(original.nodes[tid], orig_vals)
    for tid in optimized.topological_sort():
        simulate_value(optimized.nodes[tid], opt_vals)

    # Compare outputs by name (assumes optimizer preserves output names).
    max_err = 0.0
    for out_name in original.outputs:
        if out_name not in optimized.outputs:
            return False, float("inf")
        o1 = orig_vals.get(out_name, [])
        o2 = opt_vals.get(out_name, [])
        if len(o1) != len(o2):
            return False, float("inf")
        for a, b in zip(o1, o2):
            denom = max(abs(a), abs(b), 1e-6)
            max_err = max(max_err, abs(a - b) / denom)

    # Tolerance is loose because the simulator is not exact.
    return max_err < 0.5, max_err


# ---------------------------------------------------------------------------
# Cost metric
# ---------------------------------------------------------------------------

def compute_cost(
    latency: float,
    peak_mem_mb: float,
    num_nodes: int,
    correct: bool,
    latency_weight: float = 0.6,
    memory_weight: float = 0.25,
    node_weight: float = 0.15,
) -> float:
    if not correct:
        return float("inf")
    # Normalize by rough expected baseline values; these are arbitrary and
    # chosen so the baseline MLP+attention graphs score near 1.0.
    norm_latency = latency / 100.0
    norm_memory = peak_mem_mb / 500.0
    norm_nodes = num_nodes / 20.0
    return (
        latency_weight * norm_latency
        + memory_weight * norm_memory
        + node_weight * norm_nodes
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

def load_benchmarks() -> List[Graph]:
    return [
        build_simple_mlp_graph("mlp_b1_d512", batch=1, dim=512),
        build_simple_mlp_graph("mlp_b8_d512", batch=8, dim=512),
        build_attention_graph("attn_b1_s128", batch=1, seq_len=128, n_head=8, head_dim=64),
        build_attention_graph("attn_b4_s256", batch=4, seq_len=256, n_head=8, head_dim=64),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_harness(baseline: bool = False) -> Dict[str, any]:
    graphs = load_benchmarks()
    results = []
    start = time.time()

    for g in graphs:
        if baseline:
            opt_g = g.clone(f"{g.name}_baseline")
        else:
            opt_g = optimize(g.clone())

        latency = simulate_latency(opt_g)
        mem = simulate_peak_memory(opt_g)
        correct, err = check_correctness(g, opt_g)
        cost = compute_cost(latency, mem, len(opt_g.nodes), correct)

        results.append(
            {
                "graph": g.name,
                "cost": cost,
                "latency_ms": latency,
                "peak_mem_mb": mem,
                "correctness": "pass" if correct else "FAIL",
                "max_rel_err": err,
                "nodes": len(opt_g.nodes),
                "edges": sum(len(t.src) for t in opt_g.nodes.values()),
            }
        )

    # Aggregate cost is the mean across benchmarks.
    agg_cost = sum(r["cost"] for r in results) / len(results) if results else float("inf")
    agg_latency = sum(r["latency_ms"] for r in results) / len(results)
    agg_mem = sum(r["peak_mem_mb"] for r in results) / len(results)
    any_fail = any(r["correctness"] == "FAIL" for r in results)
    elapsed = time.time() - start

    summary = {
        "cost": agg_cost,
        "latency_ms": agg_latency,
        "peak_mem_mb": agg_mem,
        "correctness": "pass" if not any_fail else "FAIL",
        "benchmarks": results,
        "elapsed_s": elapsed,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="GGML optimizer harness")
    parser.add_argument("--baseline", action="store_true", help="Run identity optimizer baseline")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    summary = run_harness(baseline=args.baseline)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    mode = "baseline" if args.baseline else "optimizer"
    print(f"---")
    print(f"mode:              {mode}")
    print(f"cost:              {summary['cost']:.4f}")
    print(f"latency_ms:        {summary['latency_ms']:.2f}")
    print(f"peak_mem_mb:       {summary['peak_mem_mb']:.2f}")
    print(f"correctness:       {summary['correctness']}")
    print(f"elapsed_s:         {summary['elapsed_s']:.3f}")
    print(f"---")
    for r in summary["benchmarks"]:
        print(
            f"{r['graph']:20s} cost={r['cost']:.4f} lat={r['latency_ms']:.2f} "
            f"mem={r['peak_mem_mb']:.2f} nodes={r['nodes']} correct={r['correctness']}"
        )

    if summary["correctness"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
