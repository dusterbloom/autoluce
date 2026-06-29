"""
GGML graph data model (read-only for the agent).

This module defines a simplified Python-level representation of a GGML
computation graph. It is used by the harness and by optimizer.py.

The agent should import from this module but must not modify it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Set, Tuple


@dataclasses.dataclass(frozen=True)
class TensorId:
    """Stable identifier for a tensor/node in the graph."""
    name: str

    def __str__(self) -> str:
        return self.name


class GGMLType:
    """GGML data types."""
    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"
    Q4_0 = "Q4_0"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    I32 = "I32"


class OpType:
    """GGML operator types."""
    NONE = "NONE"
    VIEW = "VIEW"
    CPY = "CPY"
    DUP = "DUP"
    ADD = "ADD"
    MUL = "MUL"
    SUB = "SUB"
    DIV = "DIV"
    SQR = "SQR"
    SQRT = "SQRT"
    SUM = "SUM"
    MEAN = "MEAN"
    REPEAT = "REPEAT"
    CONCAT = "CONCAT"
    SILU = "SILU"
    GELU = "GELU"
    RELU = "RELU"
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"
    NORM = "NORM"
    RMS_NORM = "RMS_NORM"
    GROUP_NORM = "GROUP_NORM"
    MUL_MAT = "MUL_MAT"
    MUL_MAT_ID = "MUL_MAT_ID"
    SCALE = "SCALE"
    RESHAPE = "RESHAPE"
    VIEW = "VIEW"
    PERMUTE = "PERMUTE"
    TRANSPOSE = "TRANSPOSE"
    GET_ROWS = "GET_ROWS"
    ROPE = "ROPE"
    ALIBI = "ALIBI"
    CLAMP = "CLAMP"
    CONV_1D = "CONV_1D"
    CONV_2D = "CONV_2D"
    POOL_1D = "POOL_1D"
    POOL_2D = "POOL_2D"
    UPSCALE = "UPSCALE"
    PAD = "PAD"
    ARANGE = "ARANGE"
    TIMESTEP_EMBEDDING = "TIMESTEP_EMBEDDING"
    ARGSORT = "ARGSORT"
    TOP_K = "TOP_K"
    FLASH_ATTN_EXT = "FLASH_ATTN_EXT"
    FLASH_ATTN_BACK = "FLASH_ATTN_BACK"
    SSM_CONV = "SSM_CONV"
    SSM_SCAN = "SSM_SCAN"


# Cost weights used by the harness (kept here so optimizer.py can read them).
DEFAULT_BACKEND_COST = {
    "CPU": 1.0,
    "CUDA": 0.5,
    "METAL": 0.5,
    "VULKAN": 0.6,
    "SYCL": 0.6,
    "OPENCL": 0.8,
    "RPC": 2.0,
}


@dataclasses.dataclass
class Tensor:
    """A tensor in the GGML graph."""
    id: TensorId
    shape: Tuple[int, ...]
    dtype: str
    op: str = OpType.NONE
    src: Tuple[TensorId, ...] = dataclasses.field(default_factory=tuple)
    params: Dict[str, any] = dataclasses.field(default_factory=dict)
    backend: str = "CPU"
    is_param: bool = False
    is_view: bool = False
    buf_size: int = 0

    def __post_init__(self):
        if self.buf_size == 0 and not self.is_view:
            self.buf_size = self.nbytes()

    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_bytes(self.dtype)

    def ndim(self) -> int:
        return len(self.shape)


@dataclasses.dataclass
class Graph:
    """A GGML computation graph."""
    name: str
    nodes: Dict[TensorId, Tensor]
    inputs: List[TensorId]
    outputs: List[TensorId]

    def __post_init__(self):
        # Build use lists lazily on first access.
        self._uses: Optional[Dict[TensorId, Set[TensorId]]] = None
        self._sorted: Optional[List[TensorId]] = None

    def _build_uses(self) -> Dict[TensorId, Set[TensorId]]:
        uses: Dict[TensorId, Set[TensorId]] = {tid: set() for tid in self.nodes}
        for tid, t in self.nodes.items():
            for src in t.src:
                if src in uses:
                    uses[src].add(tid)
        return uses

    @property
    def uses(self) -> Dict[TensorId, Set[TensorId]]:
        if self._uses is None:
            self._uses = self._build_uses()
        return self._uses

    def replace_uses(self, old: TensorId, new: TensorId) -> None:
        """Replace all uses of `old` with `new`. Invalidates cached use lists."""
        for tid, t in self.nodes.items():
            if old in t.src:
                t.src = tuple(new if s == old else s for s in t.src)
        self._uses = None
        self._sorted = None

    def topological_sort(self) -> List[TensorId]:
        if self._sorted is not None:
            return list(self._sorted)
        in_degree: Dict[TensorId, int] = {tid: 0 for tid in self.nodes}
        for tid, t in self.nodes.items():
            for src in t.src:
                if src in in_degree:
                    in_degree[tid] += 1
        queue = [tid for tid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for user in self.uses.get(tid, []):
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        self._sorted = order
        return list(order)

    def reachable_from(self, roots: List[TensorId]) -> Set[TensorId]:
        """Return all nodes reachable from the given roots via forward edges."""
        visited: Set[TensorId] = set()
        stack = list(roots)
        while stack:
            tid = stack.pop()
            if tid in visited or tid not in self.nodes:
                continue
            visited.add(tid)
            for src in self.nodes[tid].src:
                stack.append(src)
        return visited

    def clone(self, new_name: Optional[str] = None) -> Graph:
        """Return a deep copy of the graph."""
        new_nodes = {tid: dataclasses.replace(t) for tid, t in self.nodes.items()}
        return Graph(
            name=new_name or self.name,
            nodes=new_nodes,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
        )


def dtype_bytes(dtype: str) -> int:
    """Return bytes per element for common GGML types (approximate for quant types)."""
    table = {
        GGMLType.F32: 4,
        GGMLType.F16: 2,
        GGMLType.BF16: 2,
        GGMLType.I32: 4,
        GGMLType.Q4_0: 0.5,
        GGMLType.Q4_K_M: 0.5625,
        GGMLType.Q5_0: 0.625,
        GGMLType.Q5_K_M: 0.6875,
        GGMLType.Q6_K: 0.75,
        GGMLType.Q8_0: 1,
    }
    return table.get(dtype, 4)


def make_tensor(
    name: str,
    shape: Tuple[int, ...],
    dtype: str = GGMLType.F32,
    op: str = OpType.NONE,
    src: Tuple[str, ...] = (),
    backend: str = "CPU",
    is_param: bool = False,
    is_view: bool = False,
    **params,
) -> Tensor:
    """Convenience factory for building tensors."""
    return Tensor(
        id=TensorId(name),
        shape=shape,
        dtype=dtype,
        op=op,
        src=tuple(TensorId(s) for s in src),
        backend=backend,
        is_param=is_param,
        is_view=is_view,
        params=params,
    )


def build_simple_mlp_graph(name: str = "mlp", batch: int = 1, dim: int = 512) -> Graph:
    """Build a tiny MLP graph for testing optimizers."""
    nodes: Dict[TensorId, Tensor] = {}

    def add(t: Tensor):
        nodes[t.id] = t

    add(make_tensor("x", (batch, dim), op=OpType.NONE))
    add(make_tensor("w1", (dim, dim * 4), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))
    add(make_tensor("b1", (dim * 4,), op=OpType.NONE, is_param=True))
    add(make_tensor("h", (batch, dim * 4), op=OpType.MUL_MAT, src=("x", "w1")))
    add(make_tensor("h_bias", (batch, dim * 4), op=OpType.ADD, src=("h", "b1")))
    add(make_tensor("h_act", (batch, dim * 4), op=OpType.SILU, src=("h_bias",)))
    add(make_tensor("w2", (dim * 4, dim), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))
    add(make_tensor("out", (batch, dim), op=OpType.MUL_MAT, src=("h_act", "w2")))

    return Graph(
        name=name,
        nodes=nodes,
        inputs=[TensorId("x")],
        outputs=[TensorId("out")],
    )


def build_attention_graph(
    name: str = "attn", batch: int = 1, seq_len: int = 128, n_head: int = 8, head_dim: int = 64
) -> Graph:
    """Build a simple attention graph for testing fusion."""
    nodes: Dict[TensorId, Tensor] = {}

    def add(t: Tensor):
        nodes[t.id] = t

    d_model = n_head * head_dim
    add(make_tensor("x", (batch, seq_len, d_model), op=OpType.NONE))
    add(make_tensor("wq", (d_model, d_model), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))
    add(make_tensor("wk", (d_model, d_model), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))
    add(make_tensor("wv", (d_model, d_model), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))
    add(make_tensor("wo", (d_model, d_model), dtype=GGMLType.Q4_0, op=OpType.NONE, is_param=True))

    add(make_tensor("q", (batch, seq_len, d_model), op=OpType.MUL_MAT, src=("x", "wq")))
    add(make_tensor("k", (batch, seq_len, d_model), op=OpType.MUL_MAT, src=("x", "wk")))
    add(make_tensor("v", (batch, seq_len, d_model), op=OpType.MUL_MAT, src=("x", "wv")))

    add(make_tensor("q_2d", (batch * seq_len, d_model), op=OpType.RESHAPE, src=("q",)))
    add(make_tensor("k_t", (d_model, batch * seq_len), op=OpType.TRANSPOSE, src=("k",)))
    add(make_tensor("qk", (batch * seq_len, batch * seq_len), op=OpType.MUL_MAT, src=("q_2d", "k_t")))
    add(make_tensor("qk_scaled", (batch * seq_len, batch * seq_len), op=OpType.SCALE, src=("qk",), scale=1.0 / math.sqrt(head_dim)))
    add(make_tensor("qk_soft", (batch * seq_len, batch * seq_len), op=OpType.SOFTMAX, src=("qk_scaled",)))
    add(make_tensor("v_2d", (batch * seq_len, d_model), op=OpType.RESHAPE, src=("v",)))
    add(make_tensor("attn_out_2d", (batch * seq_len, d_model), op=OpType.MUL_MAT, src=("qk_soft", "v_2d")))
    add(make_tensor("attn_out", (batch, seq_len, d_model), op=OpType.RESHAPE, src=("attn_out_2d",)))
    add(make_tensor("out", (batch, seq_len, d_model), op=OpType.MUL_MAT, src=("attn_out", "wo")))

    return Graph(
        name=name,
        nodes=nodes,
        inputs=[TensorId("x")],
        outputs=[TensorId("out")],
    )


def is_fusible_attention(pattern: List[Tensor]) -> bool:
    """Heuristic: detect a standard Q×K^T → scale → softmax → ×V attention."""
    if len(pattern) < 4:
        return False
    return (
        pattern[0].op == OpType.MUL_MAT
        and pattern[1].op == OpType.SCALE
        and pattern[2].op == OpType.SOFTMAX
        and pattern[3].op == OpType.MUL_MAT
    )


def print_graph(g: Graph) -> None:
    """Debug helper."""
    print(f"Graph: {g.name}")
    for tid in g.topological_sort():
        t = g.nodes[tid]
        srcs = ", ".join(str(s) for s in t.src)
        print(f"  {tid}: {t.op} [{t.dtype}] {t.shape} <- ({srcs})")
