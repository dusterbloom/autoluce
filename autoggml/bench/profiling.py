"""
Backend-aware profiler command construction for diagnostic capture.

Live capture needs the target GPU + profiler installed (nsys / rocprof); the
command building is pure and unit-tested here. Vulkan capture is layer-based
(RGP) and intentionally unsupported via this prefix wrapper.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_ENV_TO_BACKEND = {
    "GGML_CUDA": "cuda",
    "GGML_HIP": "hip",
    "GGML_VULKAN": "vulkan",
    "GGML_METAL": "metal",
}

_BACKEND_CMAKE_FLAG = {
    "cuda": "-DGGML_CUDA=ON",
    "hip": "-DGGML_HIP=ON",
    "vulkan": "-DGGML_VULKAN=ON",
    "metal": "-DGGML_METAL=ON",
}


def _probe_gpu() -> str | None:
    """Probe the host for an installed accelerator toolkit/runtime.

    Build-time detection: we look for the compiler/runtime each backend needs to
    build, not just the driver. Priority matches the profiler order: cuda > hip >
    vulkan > metal.
    """
    if shutil.which("nvcc") or os.environ.get("CUDA_HOME"):
        return "cuda"
    if shutil.which("hipcc") or Path("/opt/rocm").exists():
        return "hip"
    if shutil.which("vulkaninfo"):
        return "vulkan"
    if sys.platform == "darwin":
        return "metal"
    return None


def detect_backend(env: dict[str, str] | None = None) -> str:
    """Resolve the active ggml backend.

    Explicit env override (GGML_*=ON) wins; otherwise probe the host. Falls back
    to "cpu". Used both to pick the CMake flag (prepare/harness build) and the
    profiler wrapper (profile_command).
    """
    env = env if env is not None else os.environ
    for var, backend in _ENV_TO_BACKEND.items():
        if env.get(var) == "ON":
            return backend
    return _probe_gpu() or "cpu"


def backend_cmake_flags(backend: str) -> list[str]:
    """CMake flags that enable `backend`. Empty for cpu (no flag needed)."""
    flag = _BACKEND_CMAKE_FLAG.get(backend)
    return [flag] if flag else []


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
ROADMAP_FOR_BOUND = {
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
    return {"bound": bound, "strategy": _STRATEGY_FOR_BOUND[bound], "roadmap": ROADMAP_FOR_BOUND[bound]}
