"""
Tests for prepare.py: GPU build-decision + model discovery.

The CPU-refusal tests are Red-first (drove the extraction of should_refuse_cpu_build).
The discovery tests below are characterization: they lock the 35 GB-critical reuse
behavior of helpers that already existed when the tests were added.
"""

import pytest

from autoggml.prepare import (
    _link_into,
    build_commands,
    discover_model,
    model_search_paths,
    validate_product_backend,
)
from autoggml.bench.profiling import backend_cmake_flags
from autoggml.source_layout import SourceLayout


# --- model discovery (characterization) ----------------------------------------

def test_model_search_paths_prepends_env_and_keeps_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGGML_MODELS", str(tmp_path))
    paths = model_search_paths()
    assert paths[0] == tmp_path
    assert all(p.exists() for p in paths)


def test_model_search_paths_skips_nonexistent_env_entries(monkeypatch):
    monkeypatch.setenv("AUTOGGML_MODELS", "/definitely/does/not/exist")
    assert all(p.exists() for p in model_search_paths())


def test_discover_model_finds_existing_gguf(tmp_path, monkeypatch):
    (tmp_path / "Qwen3-1.7B-Q4_K_M.gguf").write_bytes(b"x")
    monkeypatch.setenv("AUTOGGML_MODELS", str(tmp_path))
    assert discover_model("Qwen3-1.7B-Q4_K_M.gguf") == tmp_path / "Qwen3-1.7B-Q4_K_M.gguf"


def test_discover_model_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOGGML_MODELS", str(tmp_path))
    assert discover_model("nope.gguf") is None


def test_discover_model_descends_subdirectories(tmp_path, monkeypatch):
    nested = tmp_path / "cache" / "qwen"
    nested.mkdir(parents=True)
    target = nested / "model.gguf"
    target.write_bytes(b"x")
    monkeypatch.setenv("AUTOGGML_MODELS", str(tmp_path))
    assert discover_model("model.gguf") == target


def test_link_into_makes_dest_readable(tmp_path):
    src = tmp_path / "src.gguf"
    src.write_bytes(b"payload")
    dest = tmp_path / "dest.gguf"
    _link_into(src, dest)
    assert dest.exists()
    assert dest.read_bytes() == b"payload"


# --- backend flag mapping ------------------------------------------------------

def test_backend_cmake_flags_maps_each_backend():
    assert backend_cmake_flags("cuda") == ["-DGGML_CUDA=ON"]
    assert backend_cmake_flags("vulkan") == ["-DGGML_VULKAN=ON"]


def test_backend_cmake_flags_empty_for_cpu():
    assert backend_cmake_flags("cpu") == []


def test_product_build_commands_use_server_and_real_targets(monkeypatch, tmp_path):
    checkout = tmp_path / "work" / "lucebox"
    (checkout / "server" / "deps" / "llama.cpp" / "ggml").mkdir(parents=True)
    (checkout / "server" / "CMakeLists.txt").write_text("project(dflash)\n")
    (checkout / "server" / "deps" / "llama.cpp" / "VENDOR.md").write_text("placeholder")
    monkeypatch.setenv("AUTOGGML_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    configure, build = build_commands(layout, "hip", jobs=4, use_ccache=True)

    assert configure[configure.index("-S") + 1] == str(checkout / "server")
    assert configure[configure.index("-B") + 1] == str(checkout / "build-hip")
    assert "-DDFLASH27B_GPU_BACKEND=hip" in configure
    assert "-DLLAMA_BUILD_TESTS=OFF" not in configure
    assert build[-3:] == ["dflash_server", "test_dflash", "test_deepseek4_unit"]


def test_product_backend_validation_rejects_legacy_vulkan(monkeypatch, tmp_path):
    checkout = tmp_path / "work" / "lucebox"
    monkeypatch.setenv("AUTOGGML_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    validate_product_backend(layout, "hip")
    with pytest.raises(RuntimeError, match="backend 'vulkan' is unavailable"):
        validate_product_backend(layout, "vulkan")
