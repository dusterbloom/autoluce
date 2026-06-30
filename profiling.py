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
