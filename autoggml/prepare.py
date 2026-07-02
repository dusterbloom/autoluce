"""
Read-only setup for autoggml v2.

Clones the pinned lucebox-ggml revision, downloads benchmark models,
and builds the project. Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from autoggml import ROOT
from autoggml.bench.profiling import backend_cmake_flags, detect_backend

# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------

Lucebox_GGML_URL = "https://github.com/Luce-Org/lucebox-ggml.git"
Lucebox_GGML_REF = "master"  # branch to fetch; commit is pinned dynamically

WORK_DIR = ROOT / "work"
Lucebox_DIR = WORK_DIR / "lucebox-ggml"
MODELS_DIR = WORK_DIR / "models"
BUILD_DIR = Lucebox_DIR / "build"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=capture_output)


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
    result = run(["git", "rev-parse", "HEAD"], cwd=Lucebox_DIR, check=True, capture_output=True)
    commit = result.stdout.strip()
    pin_file = WORK_DIR / "lucebox-ggml.pin"
    pin_file.write_text(commit + "\n")
    source_file = WORK_DIR / "lucebox-ggml.source"
    source_file.write_text(f"{Lucebox_GGML_URL}\n{Lucebox_GGML_REF}\n")
    print(f"Pinned lucebox-ggml: {commit}")
    # Detach at the current HEAD so experiments are deterministic.
    run(["git", "checkout", "--detach", commit], cwd=Lucebox_DIR)


def _referenced_manifest_keys() -> set[str]:
    """Manifest keys needed by the selected benchmarks.

    Honors AUTOGGML_BENCHMARKS (comma list) the same way the harness does; when
    unset, all benchmarks/*.json are considered. A manifest entry with no
    benchmark (e.g. a future model) is not downloaded."""
    selected = os.environ.get("AUTOGGML_BENCHMARKS")
    names = {b.strip() for b in selected.split(",") if b.strip()} if selected else None
    keys: set[str] = set()
    for path in sorted((ROOT / "benchmarks").glob("*.json")):
        if names is not None and path.stem not in names:
            continue
        entry = json.loads(path.read_text()).get("manifest_entry")
        if entry:
            keys.add(entry)
    return keys


# ---------------------------------------------------------------------------
# Model discovery: reuse GGUFs the user already has before hitting the network
# ---------------------------------------------------------------------------

DEFAULT_MODEL_SEARCH_PATHS = [
    "~/.cache/huggingface/hub",   # `hf_hub_download` default cache
    "~/.cache/lm-studio/models",  # LM Studio (Linux)
    "~/.lmstudio/models",         # LM Studio (alt)
    "~/models",
    "~/.local/share/models",
    "~/Downloads",
]


def model_search_paths() -> list[Path]:
    """Directories scanned for existing GGUFs. AUTOGGML_MODELS (colon-separated)
    entries are prepended; non-existent dirs are skipped."""
    env = os.environ.get("AUTOGGML_MODELS")
    raw = [p for p in (env.split(":") if env else []) if p.strip()]
    raw += DEFAULT_MODEL_SEARCH_PATHS
    paths = [Path(p).expanduser() for p in raw]
    return [p for p in paths if p.exists()]


def discover_model(local_name: str) -> Path | None:
    """Find an existing GGUF named `local_name` under the search paths. First hit wins."""
    for base in model_search_paths():
        for hit in base.rglob(local_name):
            if hit.is_file():
                return hit
    return None


def _link_into(src: Path, dest: Path) -> None:
    """Make `dest` refer to `src`: symlink preferred (zero disk), then hardlink,
    then copy. Symlink is chosen so a user deleting their cache breaks cleanly
    rather than silently retaining stale bytes."""
    try:
        dest.symlink_to(src)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dest)
        return
    except OSError:
        shutil.copy2(src, dest)


def download_models() -> None:
    """Download benchmark models referenced by the selected benchmarks."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pinned HF sources. The smoke entry is a tiny target-only model so a dev can
    # exercise the full real loop (build -> bench -> correctness -> significance)
    # with ~1 GB instead of ~35 GB: AUTOGGML_BENCHMARKS=smoke uv run prepare.py
    manifest = {
        "smoke": {
            "target": {
                "repo": "unsloth/Qwen3-1.7B-GGUF",
                "file": "Qwen3-1.7B-Q4_K_M.gguf",
                "local": "Qwen3-1.7B-Q4_K_M.gguf",
            },
        },
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

    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Model manifest written to {manifest_path}")

    wanted = _referenced_manifest_keys()
    for name, entries in manifest.items():
        if name not in wanted:
            print(f"  SKIPPED '{name}': no selected benchmark references it")
            continue
        for role, info in entries.items():
            local_path = MODELS_DIR / info["local"]
            if local_path.exists():
                print(f"  {info['local']} already exists")
                continue
            found = discover_model(info["local"])
            if found is not None:
                print(f"  Reusing {info['local']} from {found}")
                _link_into(found, local_path)
                continue
            print(f"  Downloading {info['repo']}/{info['file']} ...")
            hf_hub_download(repo_id=info["repo"], filename=info["file"], local_dir=str(MODELS_DIR))


def should_refuse_cpu_build(backend: str, env: dict[str, str] | None = None) -> bool:
    """True when a CPU build would be a silent fallback the user didn't ask for.

    The guardrail rule, stated once: refuse CPU unless AUTOGGML_ALLOW_CPU=1. Pure so
    the rule is unit-testable independent of cmake/subprocess.
    """
    env = env if env is not None else os.environ
    return backend == "cpu" and env.get("AUTOGGML_ALLOW_CPU") != "1"


def build_lucebox() -> None:
    """Configure and build lucebox-ggml for the best available accelerator."""
    if shutil.which("cmake") is None:
        print("ERROR: cmake is not installed.", file=sys.stderr)
        sys.exit(1)

    backend = detect_backend()
    if should_refuse_cpu_build(backend):
        print(
            "ERROR: no GPU toolkit detected (nvcc / hipcc / vulkaninfo / Metal).\n"
            "       A CPU build would give numbers unrelated to the GPU roadmap.\n"
            "       To force a CPU build anyway, set AUTOGGML_ALLOW_CPU=1.",
            file=sys.stderr,
        )
        sys.exit(1)
    if backend == "cpu":
        print("WARNING: building CPU-only (AUTOGGML_ALLOW_CPU=1); "
              "numbers will not reflect the GPU roadmap.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        "-G", "Ninja",
        "-S", str(Lucebox_DIR),
        "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ] + backend_cmake_flags(backend)
    print(f"Building for backend: {backend}")

    run(cmake_args)
    run(["cmake", "--build", str(BUILD_DIR), "-j", "--target", "llama-bench", "llama-server", "llama-cli"])
    print(f"Build complete: {BUILD_DIR} (backend={backend})")


def main() -> None:
    print("autoggml v2 setup")
    print("=================")
    clone_lucebox()
    download_models()
    build_lucebox()
    print("\nSetup complete. Run `uv run harness.py --baseline` next.")


if __name__ == "__main__":
    main()
