"""
Agent-editable experiment file for autoluce v2.

Modify the functions below to implement one idea per experiment.
The harness calls apply_experiment() before building and benchmarking.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from autoluce.source_layout import SourceLayout

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
PATCHES_DIR = ROOT / "patches"


def source_layout() -> SourceLayout:
    return SourceLayout.resolve(root=ROOT)


def get_cmake_flags() -> list[str]:
    """
    Return extra CMake flags for this experiment.
    Example: ["-DDFLASH27B_HIP_SM80_EQUIV=ON"]
    """
    flags = []
    # flags.append("-DGGML_CUDA=ON")
    return flags


def get_runtime_flags() -> dict[str, str]:
    """
    Return runtime flags for llama-server / llama-bench.
    Example: {"spec-draft-n-max": "15", "cache-type-k": "q8_0"}
    """
    return {
        # "spec-draft-n-max": "15",
        # "spec-draft-n-min": "1",
        # "spec-draft-p-min": "0.0",
    }


def apply_patch(patch_name: str, scope: str | None = None) -> None:
    """Apply a product- or vendor-relative patch to the Lucebox checkout."""
    patch_path = PATCHES_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")
    scope = scope or os.environ.get("AUTOLUCE_PATCH_SCOPE", "product")
    patch_root = source_layout().patch_root(scope)
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=patch_root,
        check=True,
        text=True,
    )
    print(f"Applied patch: {patch_name}")


def apply_experiment() -> dict[str, any]:
    """
    Apply the experiment to the pinned Lucebox product checkout.
    Returns a dict describing what was changed (for logging).

    Shipped as a neutral no-op baseline; the agent fills this in per experiment.
    Examples (uncomment one at a time, measure, keep or revert):

        # apply_patch("my_idea.patch")   # git-apply a file from patches/
    """
    patch_name = os.environ.get("AUTOLUCE_EXPERIMENT_PATCH")
    if patch_name:
        apply_patch(patch_name)
    return {
        "description": f"patch: {patch_name}" if patch_name else "baseline (no changes)",
        "patch": patch_name,
        "patch_scope": os.environ.get("AUTOLUCE_PATCH_SCOPE", "product"),
        "cmake_flags": get_cmake_flags(),
        "runtime_flags": get_runtime_flags(),
    }


def reset_lucebox() -> None:
    """
    Reset Lucebox to the manifest-pinned product commit.
    Called by the harness before each experiment.
    """
    layout = source_layout()
    pin_file = layout.pin_file
    if not pin_file.exists():
        raise FileNotFoundError("Run prepare.py first")
    commit = pin_file.read_text().strip()
    subprocess.run(["git", "reset", "--hard", commit], cwd=layout.checkout, check=True, text=True)
    subprocess.run(["git", "clean", "-fd", "-e", "build", "-e", "build-*"], cwd=layout.checkout, check=True, text=True)
    print(f"Reset Lucebox product to {commit}")
