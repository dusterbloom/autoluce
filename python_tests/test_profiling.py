"""
Tests for backend-aware profiler command construction.
Live capture needs the target GPU + profiler; the command building is pure here.
"""

import pytest

from autoggml.bench.profiling import classify_bottleneck, detect_backend, profile_command, summarize_rocprofv3_csv


def test_detect_backend_env_overrides_hardware(monkeypatch):
    # Hardware may report anything; an explicit env override must win.
    monkeypatch.setattr("autoggml.bench.profiling._probe_gpu", lambda: "cuda")
    assert detect_backend({"GGML_CUDA": "ON"}) == "cuda"
    assert detect_backend({"GGML_HIP": "ON"}) == "hip"
    assert detect_backend({"GGML_VULKAN": "ON"}) == "vulkan"


def test_detect_backend_probes_hardware_when_no_env(monkeypatch):
    monkeypatch.setattr("autoggml.bench.profiling._probe_gpu", lambda: "cuda")
    assert detect_backend({}) == "cuda"


def test_detect_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("autoggml.bench.profiling._probe_gpu", lambda: None)
    assert detect_backend({}) == "cpu"


def test_profile_command_cuda_wraps_with_nsys():
    out = profile_command(["llama-bench", "-m", "x"], "cuda", "/tmp/cap")
    assert out[0] == "nsys"
    assert "profile" in out
    assert "/tmp/cap" in out
    assert out[-1] == "x"


def test_profile_command_hip_wraps_with_rocprofv3():
    out = profile_command(["llama-bench"], "hip", "/tmp/cap")
    assert out[0] == "rocprofv3"
    assert "--kernel-trace" in out
    assert "--" in out
    assert out[-1] == "llama-bench"


def test_profile_command_unsupported_backend_raises():
    with pytest.raises(ValueError):
        profile_command(["llama-bench"], "vulkan", "/tmp/cap")
    with pytest.raises(ValueError):
        profile_command(["llama-bench"], "cpu", "/tmp/cap")


def test_classify_bottleneck_memory_bound_when_bandwidth_saturated():
    result = classify_bottleneck({"mem_bw_util": 0.9, "gpu_compute_util": 0.3})
    assert result["bound"] == "memory"


def test_classify_bottleneck_compute_bound_when_not_bandwidth_saturated():
    result = classify_bottleneck({"mem_bw_util": 0.4, "gpu_compute_util": 0.9})
    assert result["bound"] == "compute"


def test_classify_bottleneck_overhead_bound_when_neither_saturated():
    result = classify_bottleneck({"mem_bw_util": 0.2, "gpu_compute_util": 0.3})
    assert result["bound"] == "overhead"


def test_classify_bottleneck_memory_takes_precedence_over_compute():
    # If bandwidth is saturated, extra compute work cannot help -> memory wins.
    result = classify_bottleneck({"mem_bw_util": 0.85, "gpu_compute_util": 0.95})
    assert result["bound"] == "memory"


def test_classify_bottleneck_points_at_roadmap_items():
    result = classify_bottleneck({"mem_bw_util": 0.9, "gpu_compute_util": 0.3})
    assert result["roadmap"] and isinstance(result["roadmap"], list)


def test_summarize_rocprofv3_csv_attributes_sinkhorn_time(tmp_path):
    capture = tmp_path / "trace.csv"
    capture.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "k_bin_bcast,0,30\n"
        "reduce_rows_f32,30,50\n"
        "moe_kernel,50,100\n"
    )
    result = summarize_rocprofv3_csv(capture)
    assert result["total_dispatches"] == 3
    assert result["sinkhorn_dispatches"] == 2
    assert result["sinkhorn_duration_percent"] == pytest.approx(50.0)
