"""
Agent-editable experiment file for autoggml v2.

Modify the functions below to implement one idea per experiment.
The harness calls apply_experiment() before building and benchmarking.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
Lucebox_DIR = WORK_DIR / "lucebox-ggml"
BUILD_DIR = Lucebox_DIR / "build"
PATCHES_DIR = ROOT / "patches"


def get_cmake_flags() -> list[str]:
    """
    Return extra CMake flags for this experiment.
    Example: ["-DGGML_CUDA=ON", "-DCMAKE_CXX_FLAGS=-march=native"]
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


def apply_patch(patch_name: str) -> None:
    """Apply a patch from the patches/ directory to lucebox-ggml."""
    patch_path = PATCHES_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=Lucebox_DIR,
        check=True,
        text=True,
    )
    print(f"Applied patch: {patch_name}")


def apply_experiment() -> dict[str, any]:
    """
    Apply the experiment to lucebox-ggml.
    Returns a dict describing what was changed (for logging).
    """
    description = "baseline (no changes)"

    # -----------------------------------------------------------------------
    # Examples (uncomment one block per experiment):
    # -----------------------------------------------------------------------

    # Example 1: enable CUDA if available
    # os.environ["GGML_CUDA"] = "ON"
    # description = "enable CUDA backend"

    # Example 2: apply a code patch
    # apply_patch("0001-async-draft.patch")
    # description = "async draft generation patch"

    # Example 3: tune runtime parameters
    # flags = get_runtime_flags()
    # flags["spec-draft-n-max"] = "20"
    # description = "increase draft-n-max to 20"

    return {
        "description": description,
        "cmake_flags": get_cmake_flags(),
        "runtime_flags": get_runtime_flags(),
    }


def reset_lucebox() -> None:
    """
    Reset lucebox-ggml to the pinned commit.
    Called by the harness before each experiment.
    """
    pin_file = WORK_DIR / "lucebox-ggml.pin"
    if not pin_file.exists():
        raise FileNotFoundError("Run prepare.py first")
    commit = pin_file.read_text().strip()
    subprocess.run(["git", "reset", "--hard", commit], cwd=Lucebox_DIR, check=True, text=True)
    subprocess.run(["git", "clean", "-fd"], cwd=Lucebox_DIR, check=True, text=True)
    print(f"Reset lucebox-ggml to {commit}")
