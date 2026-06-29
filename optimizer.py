"""
GGML graph optimizer (agent-editable).

Implement optimization passes here. Each pass should be a pure function
Graph -> Graph. The harness calls optimize(graph) and measures the result.
"""

from __future__ import annotations

from typing import Dict, List, Set

from graph import Graph, OpType, Tensor, TensorId, is_fusible_attention, make_tensor


def optimize(graph: Graph) -> Graph:
    """
    Entry point. Modify the list of passes and their order to experiment.
    The default is an identity pass that returns the graph unchanged.
    """
    graph = dead_code_elimination(graph)
    # graph = fuse_silu_mul(graph)
    # graph = fuse_attention(graph)
    # graph = common_subexpression_elimination(graph)
    return graph


# ---------------------------------------------------------------------------
# Example passes (uncomment in optimize() to enable)
# ---------------------------------------------------------------------------

def dead_code_elimination(graph: Graph) -> Graph:
    """Remove nodes that are not reachable from any output."""
    reachable = graph.reachable_from(graph.outputs)
    new_nodes = {tid: t for tid, t in graph.nodes.items() if tid in reachable}
    return Graph(
        name=graph.name,
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )


def common_subexpression_elimination(graph: Graph) -> Graph:
    """Merge identical op nodes with identical inputs."""
    order = graph.topological_sort()
    canonical: Dict[tuple, TensorId] = {}
    rename: Dict[TensorId, TensorId] = {}

    for tid in order:
        t = graph.nodes[tid]
        key = (
            t.op,
            t.shape,
            t.dtype,
            tuple(rename.get(s, s) for s in t.src),
            tuple(sorted(t.params.items())),
        )
        if key in canonical:
            rename[tid] = canonical[key]
        else:
            canonical[key] = tid

    if not rename:
        return graph

    new_nodes: Dict[TensorId, Tensor] = {}
    for tid in order:
        if tid in rename:
            continue
        t = graph.nodes[tid]
        new_src = tuple(rename.get(s, s) for s in t.src)
        t.src = new_src
        new_nodes[tid] = t

    return Graph(
        name=f"{graph.name}_cse",
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )


def fuse_silu_mul(graph: Graph) -> Graph:
    """
    Detect SwiGLU-like patterns: SILU(x) * (gate) and fuse into a single op.
    Pattern: y = SILU(a); z = MUL(y, b)  ->  z = FUSED_SWIGLU(a, b)
    """
    order = graph.topological_sort()
    fusions: Dict[TensorId, TensorId] = {}

    for tid in order:
        t = graph.nodes[tid]
        if t.op != OpType.MUL or len(t.src) != 2:
            continue
        left, right = t.src
        left_t = graph.nodes.get(left)
        right_t = graph.nodes.get(right)
        if left_t and left_t.op == OpType.SILU and len(left_t.src) == 1:
            fusions[tid] = left
        elif right_t and right_t.op == OpType.SILU and len(right_t.src) == 1:
            fusions[tid] = right

    if not fusions:
        return graph

    new_nodes: Dict[TensorId, Tensor] = {}
    for tid in order:
        t = graph.nodes[tid]
        if tid in fusions:
            # Replace the MUL with a fused op.
            silu_tid = fusions[tid]
            silu_t = graph.nodes[silu_tid]
            gate_tid = t.src[1] if t.src[0] == silu_tid else t.src[0]
            fused = make_tensor(
                name=f"{t.id.name}_fused_swiglu",
                shape=t.shape,
                dtype=t.dtype,
                op="FUSED_SWIGLU",
                src=(silu_t.src[0].name, gate_tid.name),
                backend=t.backend,
            )
            new_nodes[fused.id] = fused
            graph.replace_uses(tid, fused.id)
        elif tid not in {fusions[v] for v in fusions}:
            new_nodes[tid] = t

    return Graph(
        name=f"{graph.name}_swiglu",
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )


def fuse_attention(graph: Graph) -> Graph:
    """
    Detect a simple attention pattern and replace it with FLASH_ATTN_EXT.
    This is intentionally conservative; only exact patterns are replaced.
    """
    order = graph.topological_sort()
    to_remove: Set[TensorId] = set()
    replacements: Dict[TensorId, TensorId] = {}

    for tid in order:
        t = graph.nodes[tid]
        if t.op != OpType.MUL_MAT or len(t.src) != 2:
            continue
        # Look for QK^T -> scale -> softmax -> *V chain.
        qk_tid = tid
        qk_t = t
        if qk_t.op != OpType.MUL_MAT:
            continue
        users = [u for u in order if qk_tid in graph.nodes[u].src]
        if len(users) != 1 or graph.nodes[users[0]].op != OpType.SCALE:
            continue
        scale_tid = users[0]
        users2 = [u for u in order if scale_tid in graph.nodes[u].src]
        if len(users2) != 1 or graph.nodes[users2[0]].op != OpType.SOFTMAX:
            continue
        soft_tid = users2[0]
        users3 = [u for u in order if soft_tid in graph.nodes[u].src]
        if len(users3) != 1 or graph.nodes[users3[0]].op != OpType.MUL_MAT:
            continue
        out_tid = users3[0]

        # Identify q, k, v from the chain.
        q_src = qk_t.src[0]
        k_src = qk_t.src[1]
        v_src = graph.nodes[out_tid].src[1]

        fused = make_tensor(
            name=f"{out_tid.name}_flash",
            shape=graph.nodes[out_tid].shape,
            dtype=graph.nodes[out_tid].dtype,
            op=OpType.FLASH_ATTN_EXT,
            src=(q_src.name, k_src.name, v_src.name),
            backend="CUDA",
        )
        replacements[out_tid] = fused.id
        to_remove.update({qk_tid, scale_tid, soft_tid})

    if not replacements:
        return graph

    new_nodes: Dict[TensorId, Tensor] = {}
    for tid in order:
        t = graph.nodes[tid]
        if tid in to_remove or tid in replacements:
            continue
        # Apply pending replacements to source lists.
        new_src = []
        for s in t.src:
            if s in replacements:
                new_src.append(replacements[s])
            else:
                new_src.append(s)
        t.src = tuple(new_src)
        new_nodes[tid] = t

    for old_tid, fused_id in replacements.items():
        # Build the fused node from the template we created earlier.
        # We recreate it with the resolved source names.
        old_out = graph.nodes[old_tid]
        q, k, v = old_out.src
        fused = make_tensor(
            name=fused_id.name,
            shape=old_out.shape,
            dtype=old_out.dtype,
            op=OpType.FLASH_ATTN_EXT,
            src=(q.name, k.name, v.name),
            backend="CUDA",
        )
        new_nodes[fused.id] = fused

    return Graph(
        name=f"{graph.name}_flash",
        nodes=new_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )
