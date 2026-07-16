"""
Tests for prepare.py: GPU build-decision + model discovery.

The CPU-refusal tests are Red-first (drove the extraction of should_refuse_cpu_build).
The discovery tests below are characterization: they lock the 35 GB-critical reuse
behavior of helpers that already existed when the tests were added.
"""

import pytest

from autoluce.models import load_catalog
from autoluce.prepare import (
    _catalog_artifact_manifest,
    _link_into,
    _validate_model_artifact,
    build_commands,
    download_models,
    discover_model,
    model_search_paths,
    validate_product_backend,
)
from autoluce.bench.profiling import backend_cmake_flags
from autoluce.source_layout import SourceLayout


# --- model discovery (characterization) ----------------------------------------

def test_model_search_paths_prepends_env_and_keeps_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLUCE_MODELS", str(tmp_path))
    paths = model_search_paths()
    assert paths[0] == tmp_path
    assert all(p.exists() for p in paths)


def test_model_search_paths_skips_nonexistent_env_entries(monkeypatch):
    monkeypatch.setenv("AUTOLUCE_MODELS", "/definitely/does/not/exist")
    assert all(p.exists() for p in model_search_paths())


def test_discover_model_finds_existing_gguf(tmp_path, monkeypatch):
    (tmp_path / "Qwen3-1.7B-Q4_K_M.gguf").write_bytes(b"x")
    monkeypatch.setenv("AUTOLUCE_MODELS", str(tmp_path))
    assert discover_model("Qwen3-1.7B-Q4_K_M.gguf") == tmp_path / "Qwen3-1.7B-Q4_K_M.gguf"


def test_discover_model_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLUCE_MODELS", str(tmp_path))
    assert discover_model("nope.gguf") is None


def test_discover_model_descends_subdirectories(tmp_path, monkeypatch):
    nested = tmp_path / "cache" / "qwen"
    nested.mkdir(parents=True)
    target = nested / "model.gguf"
    target.write_bytes(b"x")
    monkeypatch.setenv("AUTOLUCE_MODELS", str(tmp_path))
    assert discover_model("model.gguf") == target


def test_link_into_makes_dest_readable(tmp_path):
    src = tmp_path / "src.gguf"
    src.write_bytes(b"payload")
    dest = tmp_path / "dest.gguf"
    _link_into(src, dest)
    assert dest.exists()
    assert dest.read_bytes() == b"payload"


def test_bonsai_27b_catalog_entry_pins_public_q1_artifact():
    entry = load_catalog()["bonsai-27b-q1"]

    assert entry.quant == "Q1_0_g128"
    assert entry.files == ["Bonsai-27B-Q1_0.gguf"]
    assert entry.expected_size_bytes == 3_803_452_480
    assert entry.path_env == "AUTOLUCE_BONSAI_27B_MODEL"
    assert entry.repository == "prism-ml/Bonsai-27B-gguf"
    assert entry.revision == "0cf7e3d21581b169b4df1de8bf01316000e2fbb7"
    assert entry.sha256 == "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
    assert entry.metadata == {
        "architecture": "qwen35",
        "base_model": "Qwen/Qwen3.6-27B",
        "parameters": 26_895_998_464,
        "context_length": 262_144,
        "vocab_size": 248_320,
        "vision_projector": "optional",
        "campaign_scope": "text-only",
    }


def test_bonsai_27b_catalog_entry_pins_public_dspark_artifact():
    entry = load_catalog()["bonsai-27b-dspark-q4_1"]

    assert entry.quant == "Q4_1"
    assert entry.files == ["Bonsai-27B-dspark-Q4_1.gguf"]
    assert entry.expected_size_bytes == 1_787_468_768
    assert entry.path_env == "AUTOLUCE_BONSAI_27B_DRAFT"
    assert entry.repository == "prism-ml/Bonsai-27B-gguf"
    assert entry.revision == "0cf7e3d21581b169b4df1de8bf01316000e2fbb7"
    assert entry.sha256 == "25e73f9f7ab5d1f7f1336711496dbc12da674e639ec88d579dc8683045befb1b"
    assert entry.metadata == {
        "architecture": "dspark",
        "target": "bonsai-27b-q1",
        "block_size": 4,
        "context_length": 4096,
        "vocab_size": 248_320,
        "mask_token_id": 248_319,
        "markov_rank": 256,
        "target_layers": [1, 16, 31, 46, 61],
    }


def test_catalog_target_manifest_honors_external_override(monkeypatch, tmp_path):
    entry = load_catalog()["bonsai-27b-q1"]
    external = tmp_path / entry.first_file
    monkeypatch.setenv(entry.path_env, str(external))

    target = _catalog_artifact_manifest(entry)

    assert target["path"] == str(external)
    assert target["files"] == [str(external)]
    assert "repo" not in target
    assert target["sha256"] == entry.sha256


def test_download_models_resolves_public_bonsai_target(monkeypatch, tmp_path):
    from autoluce import prepare

    downloads = []
    validations = []

    def fake_download(*, repo_id, filename, local_dir, revision):
        downloads.append((repo_id, filename, local_dir, revision))
        return str(tmp_path / filename)

    monkeypatch.setenv("AUTOLUCE_BENCHMARKS", "bonsai27b-q1-target")
    monkeypatch.setattr(prepare, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(prepare, "discover_model", lambda _name: None)
    monkeypatch.setattr(prepare, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        prepare,
        "_validate_model_artifact",
        lambda path, expected_size, sha256: validations.append((path, expected_size, sha256)),
    )

    download_models()

    assert downloads == [(
        "prism-ml/Bonsai-27B-gguf",
        "Bonsai-27B-Q1_0.gguf",
        str(tmp_path),
        "0cf7e3d21581b169b4df1de8bf01316000e2fbb7",
    )]
    assert validations == [(
        tmp_path / "Bonsai-27B-Q1_0.gguf",
        3_803_452_480,
        "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
    )]


def test_download_models_resolves_public_bonsai_pair(monkeypatch, tmp_path):
    from autoluce import prepare

    downloads = []

    def fake_download(*, repo_id, filename, local_dir, revision):
        downloads.append((repo_id, filename, local_dir, revision))
        return str(tmp_path / filename)

    monkeypatch.setenv("AUTOLUCE_BENCHMARKS", "bonsai27b-q1-dspark")
    monkeypatch.setattr(prepare, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(prepare, "discover_model", lambda _name: None)
    monkeypatch.setattr(prepare, "hf_hub_download", fake_download)
    monkeypatch.setattr(prepare, "_validate_model_artifact", lambda *_args: None)

    download_models()

    assert downloads == [
        (
            "prism-ml/Bonsai-27B-gguf",
            "Bonsai-27B-Q1_0.gguf",
            str(tmp_path),
            "0cf7e3d21581b169b4df1de8bf01316000e2fbb7",
        ),
        (
            "prism-ml/Bonsai-27B-gguf",
            "Bonsai-27B-dspark-Q4_1.gguf",
            str(tmp_path),
            "0cf7e3d21581b169b4df1de8bf01316000e2fbb7",
        ),
    ]


def test_validate_model_artifact_checks_size_and_sha256(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"bonsai")

    _validate_model_artifact(
        artifact,
        expected_size=6,
        sha256="7dd7122ad9bf240f04fdf988a0df4a2552098ad8ed8df429bed1056ebdb64387",
    )

    with pytest.raises(ValueError, match="size mismatch"):
        _validate_model_artifact(artifact, expected_size=7, sha256=None)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_model_artifact(artifact, expected_size=6, sha256="0" * 64)


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
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    configure, build = build_commands(layout, "hip", jobs=4, use_ccache=True)

    assert configure[configure.index("-S") + 1] == str(checkout / "server")
    assert configure[configure.index("-B") + 1] == str(checkout / "build-hip")
    assert "-DDFLASH27B_GPU_BACKEND=hip" in configure
    assert "-DLLAMA_BUILD_TESTS=OFF" not in configure
    assert build[-3:] == ["dflash_server", "test_dflash", "test_deepseek4_unit"]


def test_product_backend_validation_rejects_legacy_vulkan(monkeypatch, tmp_path):
    checkout = tmp_path / "work" / "lucebox"
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    validate_product_backend(layout, "hip")
    with pytest.raises(RuntimeError, match="backend 'vulkan' is unavailable"):
        validate_product_backend(layout, "vulkan")
