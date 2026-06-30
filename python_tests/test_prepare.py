"""
Tests for prepare.py: GPU build-decision + model discovery.

The CPU-refusal tests are Red-first (drove the extraction of should_refuse_cpu_build).
The discovery tests below are characterization: they lock the 35 GB-critical reuse
behavior of helpers that already existed when the tests were added.
"""

from prepare import _link_into, discover_model, model_search_paths, should_refuse_cpu_build
from profiling import backend_cmake_flags


# --- CPU-refusal decision (pure; Red-first) ------------------------------------

def test_should_refuse_cpu_build_when_cpu_and_no_opt_in():
    assert should_refuse_cpu_build("cpu", {}) is True


def test_should_not_refuse_cpu_when_opted_in():
    assert should_refuse_cpu_build("cpu", {"AUTOGGML_ALLOW_CPU": "1"}) is False


def test_should_not_refuse_cpu_when_gpu():
    assert should_refuse_cpu_build("cuda", {}) is False
    assert should_refuse_cpu_build("vulkan", {}) is False


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
