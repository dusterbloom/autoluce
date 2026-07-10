"""Manifest-driven Lucebox product/vendor boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autoluce.source_layout import (
    SourceDrift,
    SourceLayout,
    SourceManifest,
    check_remote_drift,
    parse_vendor_manifest,
)


def _hub_checkout(root: Path) -> Path:
    checkout = root / "work" / "lucebox"
    (checkout / "server" / "deps" / "llama.cpp" / "ggml").mkdir(parents=True)
    (checkout / "server" / "CMakeLists.txt").write_text("project(dflash)\n")
    (checkout / "server" / "deps" / "llama.cpp" / "VENDOR.md").write_text(
        "# Vendored llama.cpp/ggml snapshot\n"
        "- Source repository: https://github.com/Luce-Org/lucebox-ggml\n"
        "- Source base branch: `luce-dflash`\n"
        "- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`\n"
        "- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)\n"
        "- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)\n"
    )
    return checkout


def test_repository_manifest_pins_current_vendored_lucebox_product():
    manifest = SourceManifest.load()

    assert manifest.repository == "https://github.com/Luce-Org/lucebox-hub.git"
    assert manifest.ref == "5e302cbb483819cd21e72f5dd8becaa609eca8cf"
    assert manifest.track == "refs/heads/main"
    assert manifest.layout == "lucebox-hub-vendored"
    assert manifest.vendor_subdir == "server/deps/llama.cpp"
    assert manifest.submodules_by_backend == {
        "cuda": ["server/deps/Block-Sparse-Attention"],
        "hip": [],
    }


def test_repository_normalization_does_not_strip_valid_name_characters(monkeypatch, tmp_path):
    checkout = _hub_checkout(tmp_path)
    vendor_manifest = checkout / "server" / "deps" / "llama.cpp" / "VENDOR.md"
    vendor_manifest.write_text(vendor_manifest.read_text().replace("lucebox-ggml", "lucebox-ggml.git"))
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))

    assert SourceLayout.resolve(root=tmp_path).validate_vendor_provenance().source_branch == "luce-dflash"


def test_layout_has_one_authoritative_path_map(monkeypatch, tmp_path):
    checkout = _hub_checkout(tmp_path)
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    assert layout.checkout == checkout
    assert layout.cmake_source == checkout / "server"
    assert layout.vendor_root == checkout / "server" / "deps" / "llama.cpp"
    assert layout.ggml_root == layout.vendor_root / "ggml"
    assert layout.build_dir("hip") == checkout / "build-hip"
    assert layout.binary("dflash_server", "hip") == checkout / "build-hip" / "dflash_server"
    assert layout.patch_root("product") == checkout
    assert layout.patch_root("vendor") == layout.vendor_root
    assert layout.pin_file == tmp_path / "work" / "lucebox.pin"
    layout.require_capability("product-build")
    with pytest.raises(RuntimeError, match="does not provide 'llama-tools'"):
        layout.require_capability("llama-tools")


def test_layout_detects_hub_and_refuses_wrong_or_missing_shape(monkeypatch, tmp_path):
    checkout = _hub_checkout(tmp_path)
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    assert SourceLayout.resolve(root=tmp_path).detect() == "lucebox-hub-vendored"

    (checkout / "server" / "deps" / "llama.cpp" / "VENDOR.md").unlink()
    with pytest.raises(ValueError, match="VENDOR.md"):
        SourceLayout.resolve(root=tmp_path).validate()


def test_backend_flags_and_targets_belong_to_layout(monkeypatch, tmp_path):
    checkout = _hub_checkout(tmp_path)
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)

    assert layout.cmake_backend_flags("cuda") == ["-DDFLASH27B_GPU_BACKEND=cuda"]
    assert layout.cmake_backend_flags("hip") == ["-DDFLASH27B_GPU_BACKEND=hip"]
    assert layout.submodules("cuda") == ["server/deps/Block-Sparse-Attention"]
    assert layout.submodules("hip") == []
    assert layout.build_targets == ["dflash_server", "test_dflash", "test_deepseek4_unit"]
    with pytest.raises(ValueError, match="does not support backend 'vulkan'"):
        layout.cmake_backend_flags("vulkan")


def test_vendor_manifest_is_machine_readable_and_checked(monkeypatch, tmp_path):
    checkout = _hub_checkout(tmp_path)
    monkeypatch.setenv("AUTOLUCE_SOURCE_ROOT", str(checkout))
    layout = SourceLayout.resolve(root=tmp_path)
    provenance = parse_vendor_manifest(layout.vendor_manifest_path)

    assert provenance.source_branch == "luce-dflash"
    assert provenance.source_commit == "6fbe72d67069136bbd370be703e1d4f441b5e942"
    assert provenance.included_commits == [
        "0fe65d9354b7c5da52a7741d2e37ba85f0d0c925",
        "0699be81480428f01b9b7ac49a09a2d51c77f8df",
    ]
    assert layout.validate_vendor_provenance().source_commit == provenance.source_commit


class _RemoteRunner:
    def __init__(self, sha: str):
        self.sha = sha

    def __call__(self, args, **kwargs):
        return subprocess.CompletedProcess(args, 0, f"{self.sha}\trefs/heads/main\n", "")


def test_remote_drift_check_is_pure_and_reports_update():
    manifest = SourceManifest.load()
    current = check_remote_drift(manifest, runner=_RemoteRunner(manifest.ref))
    changed = check_remote_drift(manifest, runner=_RemoteRunner("a" * 40))

    assert current == SourceDrift(manifest.ref, manifest.ref, False)
    assert changed == SourceDrift(manifest.ref, "a" * 40, True)
