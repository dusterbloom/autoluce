"""Manifest-driven ownership boundary for Lucebox and its vendored ggml."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from autoggml import ROOT


DEFAULT_MANIFEST = ROOT / "sources" / "lucebox.toml"


def _canonical_repository(value: str) -> str:
    return value.removesuffix(".git").rstrip("/")


@dataclass(frozen=True)
class VendorExpectation:
    repository: str
    branch: str
    base_commit: str
    included_commits: list[str]


@dataclass(frozen=True)
class SourceManifest:
    schema_version: int
    name: str
    repository: str
    ref: str
    track: str
    layout: str
    checkout_subdir: str
    pin_subpath: str
    source_record_subpath: str
    cmake_source_subdir: str
    vendor_subdir: str
    vendor_manifest_subpath: str
    supported_backends: list[str]
    build_targets: list[str]
    submodules_by_backend: dict[str, list[str]]
    runtime: str
    capabilities: list[str]
    vendor: VendorExpectation

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST) -> "SourceManifest":
        raw = tomllib.loads(path.read_text())
        raw["vendor"] = VendorExpectation(**raw["vendor"])
        manifest = cls(**raw)
        if manifest.schema_version != 1:
            raise ValueError(f"unsupported source manifest schema: {manifest.schema_version}")
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.ref):
            raise ValueError("source manifest ref must be a full Git commit")
        if not manifest.track.startswith("refs/heads/"):
            raise ValueError("source manifest track must name a branch ref")
        if len(set(manifest.supported_backends)) != len(manifest.supported_backends):
            raise ValueError("source manifest backends must be unique")
        if set(manifest.submodules_by_backend) - set(manifest.supported_backends):
            raise ValueError("source manifest submodule backends must be supported")
        submodules = [path for paths in manifest.submodules_by_backend.values() for path in paths]
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in submodules):
            raise ValueError("source manifest submodules must be checkout-relative")
        return manifest


@dataclass(frozen=True)
class VendorProvenance:
    source_repository: str
    source_branch: str
    source_commit: str
    included_commits: list[str]


@dataclass(frozen=True)
class SourceDrift:
    pinned: str
    upstream: str
    changed: bool


def parse_vendor_manifest(path: Path) -> VendorProvenance:
    text = path.read_text()

    def value(label: str) -> str:
        match = re.search(rf"^- {re.escape(label)}:\s*(.+?)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"VENDOR.md is missing '{label}'")
        return match.group(1).strip().strip("`")

    commits = re.findall(r"^- Included (?:merged|test) PR:.*?\(`([0-9a-f]{40})`\)", text, re.MULTILINE)
    return VendorProvenance(
        source_repository=value("Source repository"),
        source_branch=value("Source base branch"),
        source_commit=value("Source base commit"),
        included_commits=commits,
    )


@dataclass(frozen=True)
class SourceLayout:
    root: Path
    manifest: SourceManifest
    checkout: Path

    @classmethod
    def resolve(cls, root: Path = ROOT, manifest_path: Path = DEFAULT_MANIFEST) -> "SourceLayout":
        manifest = SourceManifest.load(manifest_path)
        override = os.environ.get("AUTOGGML_SOURCE_ROOT")
        checkout = Path(override).expanduser() if override else root / manifest.checkout_subdir
        return cls(root, manifest, checkout)

    @property
    def pin_file(self) -> Path:
        return self.root / self.manifest.pin_subpath

    @property
    def source_record_file(self) -> Path:
        return self.root / self.manifest.source_record_subpath

    @property
    def cmake_source(self) -> Path:
        return self.checkout / self.manifest.cmake_source_subdir

    @property
    def vendor_root(self) -> Path:
        return self.checkout / self.manifest.vendor_subdir

    @property
    def ggml_root(self) -> Path:
        return self.vendor_root / "ggml"

    @property
    def vendor_manifest_path(self) -> Path:
        return self.checkout / self.manifest.vendor_manifest_subpath

    @property
    def build_targets(self) -> list[str]:
        return list(self.manifest.build_targets)

    @property
    def runtime(self) -> str:
        return self.manifest.runtime

    def require_capability(self, capability: str) -> None:
        if capability not in self.manifest.capabilities:
            raise RuntimeError(
                f"Lucebox source layout '{self.manifest.layout}' does not provide '{capability}'. "
                "Run `uv run autoggml source status` and use a product-runtime adapter."
            )

    def build_dir(self, backend: str) -> Path:
        subdir = os.environ.get("AUTOGGML_BUILD_SUBDIR", f"build-{backend}")
        return self.checkout / subdir

    def binary(self, target: str, backend: str) -> Path:
        if target not in self.build_targets:
            raise ValueError(f"unknown Lucebox build target: {target}")
        return self.build_dir(backend) / target

    def patch_root(self, scope: str) -> Path:
        if scope == "product":
            return self.checkout
        if scope == "vendor":
            return self.vendor_root
        raise ValueError("patch scope must be 'product' or 'vendor'")

    def cmake_backend_flags(self, backend: str) -> list[str]:
        if backend not in self.manifest.supported_backends:
            raise ValueError(
                f"source layout '{self.manifest.layout}' does not support backend '{backend}'; "
                f"supported: {', '.join(self.manifest.supported_backends)}"
            )
        return [f"-DDFLASH27B_GPU_BACKEND={backend}"]

    def submodules(self, backend: str) -> list[str]:
        if backend not in self.manifest.supported_backends:
            raise ValueError(f"source layout '{self.manifest.layout}' does not support backend '{backend}'")
        return list(self.manifest.submodules_by_backend.get(backend, []))

    def detect(self) -> str:
        if (
            (self.cmake_source / "CMakeLists.txt").is_file()
            and (self.vendor_root / "ggml").is_dir()
            and self.vendor_manifest_path.is_file()
        ):
            return "lucebox-hub-vendored"
        return "unknown"

    def validate_vendor_provenance(self) -> VendorProvenance:
        provenance = parse_vendor_manifest(self.vendor_manifest_path)
        expected = self.manifest.vendor
        mismatches = []
        if _canonical_repository(provenance.source_repository) != _canonical_repository(expected.repository):
            mismatches.append("repository")
        if provenance.source_branch != expected.branch:
            mismatches.append("branch")
        if provenance.source_commit != expected.base_commit:
            mismatches.append("base_commit")
        if provenance.included_commits != expected.included_commits:
            mismatches.append("included_commits")
        if mismatches:
            raise ValueError("vendored ggml provenance drift: " + ", ".join(mismatches))
        return provenance

    def validate(self) -> VendorProvenance:
        detected = self.detect()
        if detected != self.manifest.layout:
            missing = "VENDOR.md" if not self.vendor_manifest_path.is_file() else self.manifest.layout
            raise ValueError(f"source checkout does not match {self.manifest.layout}: missing or invalid {missing}")
        return self.validate_vendor_provenance()


def check_remote_drift(
    manifest: SourceManifest,
    runner: Callable = subprocess.run,
) -> SourceDrift:
    process = runner(
        ["git", "ls-remote", manifest.repository, manifest.track],
        capture_output=True, text=True, check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "could not query Lucebox upstream")
    line = next((line for line in process.stdout.splitlines() if line.strip()), "")
    if not line:
        raise RuntimeError(f"upstream ref not found: {manifest.track}")
    upstream = line.split()[0]
    return SourceDrift(manifest.ref, upstream, upstream != manifest.ref)
