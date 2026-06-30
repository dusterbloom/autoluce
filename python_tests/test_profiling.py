"""
Tests for backend-aware profiler command construction.
Live capture needs the target GPU + profiler; the command building is pure here.
"""

import pytest

from profiling import detect_backend, profile_command


def test_detect_backend_from_env():
    assert detect_backend({"GGML_CUDA": "ON"}) == "cuda"
    assert detect_backend({"GGML_HIP": "ON"}) == "hip"
    assert detect_backend({"GGML_VULKAN": "ON"}) == "vulkan"
    assert detect_backend({}) == "cpu"


def test_profile_command_cuda_wraps_with_nsys():
    out = profile_command(["llama-bench", "-m", "x"], "cuda", "/tmp/cap")
    assert out[0] == "nsys"
    assert "profile" in out
    assert "/tmp/cap" in out
    assert out[-1] == "x"


def test_profile_command_hip_wraps_with_rocprof():
    out = profile_command(["llama-bench"], "hip", "/tmp/cap")
    assert out[0] == "rocprof"
    assert out[-1] == "llama-bench"


def test_profile_command_unsupported_backend_raises():
    with pytest.raises(ValueError):
        profile_command(["llama-bench"], "vulkan", "/tmp/cap")
    with pytest.raises(ValueError):
        profile_command(["llama-bench"], "cpu", "/tmp/cap")
