"""
Backend-aware profiler command construction for diagnostic capture.

Live capture needs the target GPU + profiler installed (nsys / rocprof); the
command building is pure and unit-tested here. Vulkan capture is layer-based
(RGP) and intentionally unsupported via this prefix wrapper.
"""

from __future__ import annotations

import os


def detect_backend(env: dict[str, str] | None = None) -> str:
    """Infer the active ggml backend from build/runtime env (cuda > hip > vulkan > cpu)."""
    env = env if env is not None else os.environ
    if env.get("GGML_CUDA") == "ON":
        return "cuda"
    if env.get("GGML_HIP") == "ON":
        return "hip"
    if env.get("GGML_VULKAN") == "ON":
        return "vulkan"
    return "cpu"


def profile_command(cmd: list[str], backend: str, output_path: str) -> list[str]:
    """Wrap a benchmark command with the backend's profiler, writing to output_path."""
    if backend == "cuda":
        return ["nsys", "profile", "--stats=true", "--force-overwrite=true", "-o", output_path, *cmd]
    if backend == "hip":
        return ["rocprof", "--stats", "-o", output_path, *cmd]
    raise ValueError(
        f"no profiler configured for backend '{backend}' "
        "(cuda -> nsys, hip -> rocprof; vulkan uses RGP layers, not a prefix wrapper)"
    )


# Throughput thresholds (fractions of peak) at which a resource is "saturated".
_BANDWIDTH_SATURATED = 0.80
_COMPUTE_SATURATED = 0.80

# Roadmap item numbers to pursue for each bound (see ROADMAP.md).
_ROADMAP_FOR_BOUND = {
    "memory": [10, 11, 14],   # KV O(1) rollback + shared prefix, draft quant, KV quant sweep
    "compute": [9, 12],       # fused draft subgraph, Q4_K verify kernel tuning
    "overhead": [8, 9],       # CUDA-graph verify, fused draft subgraph
}
_STRATEGY_FOR_BOUND = {
    "memory": "reduce data movement (KV quant, shared-prefix compute, fusion)",
    "compute": "do less compute work per token (kernel fusion, better algorithms)",
    "overhead": "cut dispatch/launch overhead (CUDA graphs, fused draft subgraph)",
}


def classify_bottleneck(summary: dict[str, float]) -> dict:
    """
    Turn a profile summary into a bottleneck verdict + strategy.

    `summary` carries resource utilizations as fractions of peak, e.g.
    {"mem_bw_util": 0.9, "gpu_compute_util": 0.3}. Memory saturation takes
    precedence: if bandwidth is the wall, compute work cannot help. Returns
    {"bound", "strategy", "roadmap"} where roadmap lists ROADMAP.md item numbers.
    """
    bound = (
        "memory" if summary.get("mem_bw_util", 0.0) >= _BANDWIDTH_SATURATED
        else "compute" if summary.get("gpu_compute_util", 0.0) >= _COMPUTE_SATURATED
        else "overhead"
    )
    return {"bound": bound, "strategy": _STRATEGY_FOR_BOUND[bound], "roadmap": _ROADMAP_FOR_BOUND[bound]}
