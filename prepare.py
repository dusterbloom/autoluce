"""
Read-only setup for autoggml v2.

Clones the pinned lucebox-ggml revision, downloads benchmark models,
and builds the project. Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------

Lucebox_GGML_URL = "https://github.com/Luce-Org/lucebox-ggml.git"
Lucebox_GGML_REF = "master"  # branch to fetch; commit is pinned dynamically

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
Lucebox_DIR = WORK_DIR / "lucebox-ggml"
MODELS_DIR = WORK_DIR / "models"
BUILD_DIR = Lucebox_DIR / "build"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------


def clone_lucebox() -> None:
    """Clone or update lucebox-ggml to the pinned ref."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not Lucebox_DIR.exists():
        run(["git", "clone", "--depth", "100", "--branch", Lucebox_GGML_REF, Lucebox_GGML_URL, str(Lucebox_DIR)])
    else:
        run(["git", "fetch", "origin", Lucebox_GGML_REF], cwd=Lucebox_DIR)
    # Resolve the current HEAD commit hash and store it.
    result = run(["git", "rev-parse", "HEAD"], cwd=Lucebox_DIR, check=True)
    commit = result.stdout.strip()
    pin_file = WORK_DIR / "lucebox-ggml.pin"
    pin_file.write_text(commit + "\n")
    print(f"Pinned lucebox-ggml: {commit}")
    # Detach at the current HEAD so experiments are deterministic.
    run(["git", "checkout", "--detach", commit], cwd=Lucebox_DIR)


def download_models() -> None:
    """Download benchmark models if not present and verify hashes."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model files used by the Lucebox DFlash server, pinned to specific HF sources.
    manifest = {
        "qwen36-27b": {
            "target": {
                "repo": "unsloth/Qwen3.6-27B-GGUF",
                "file": "Qwen3.6-27B-Q4_K_M.gguf",
                "local": "Qwen3.6-27B-Q4_K_M.gguf",
            },
            "draft": {
                "repo": "Lucebox/Qwen3.6-27B-DFlash-GGUF",
                "file": "dflash-draft-3.6-q4_k_m.gguf",
                "local": "dflash-draft-3.6-q4_k_m.gguf",
            },
        },
        "gemma4-26b-a4b": {
            "target": {
                "repo": "bartowski/gemma-4-26B-A4B-it-GGUF",
                "file": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                "local": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
            },
            "draft": {
                "repo": "Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF",
                "file": "gemma-4-26B-A4B-it-DFlash.gguf",
                "local": "gemma-4-26B-A4B-it-DFlash.gguf",
            },
        },
    }

    import json
    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Model manifest written to {manifest_path}")

    # Try to download using huggingface-cli if available.
    hf_cli = shutil.which("huggingface-cli")
    for name, entries in manifest.items():
        for role, info in entries.items():
            local_path = MODELS_DIR / info["local"]
            if local_path.exists():
                print(f"  {info['local']} already exists")
                continue
            if hf_cli:
                print(f"  Downloading {info['repo']}/{info['file']} ...")
                try:
                    subprocess.run([
                        hf_cli, "download", info["repo"], info["file"],
                        "--local-dir", str(MODELS_DIR)
                    ], check=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR downloading {info['local']}: {e}")
            else:
                print(f"  SKIPPED {info['local']}: huggingface-cli not installed")


def build_lucebox() -> None:
    """Configure and build lucebox-ggml."""
    if shutil.which("cmake") is None:
        print("ERROR: cmake is not installed.", file=sys.stderr)
        sys.exit(1)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # Default CPU build. For GPU, set GGML_CUDA=ON or GGML_METAL=ON via environment.
    cmake_args = [
        "cmake",
        "-S", str(Lucebox_DIR),
        "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
    ]
    if os.environ.get("GGML_CUDA") == "ON":
        cmake_args.append("-DGGML_CUDA=ON")
    if os.environ.get("GGML_METAL") == "ON":
        cmake_args.append("-DGGML_METAL=ON")

    run(cmake_args)
    run(["cmake", "--build", str(BUILD_DIR), "-j", "--target", "llama-bench", "llama-server", "llama-cli"])
    print(f"Build complete: {BUILD_DIR}")


def main() -> None:
    print("autoggml v2 setup")
    print("=================")
    clone_lucebox()
    download_models()
    build_lucebox()
    print("\nSetup complete. Run `uv run harness.py --baseline` next.")


if __name__ == "__main__":
    main()
