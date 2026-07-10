"""
Read-only setup for autoluce v2.

Clones the pinned Lucebox product revision, downloads benchmark models,
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

from autoluce import ROOT
from autoluce.bench.profiling import detect_backend
from autoluce.models import load_catalog, resolve_entry_files
from autoluce.source_layout import SourceLayout

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK_DIR = ROOT / "work"
MODELS_DIR = WORK_DIR / "models"

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
    """Clone or update the manifest-pinned Lucebox product checkout."""
    layout = SourceLayout.resolve()
    manifest = layout.manifest
    repository = os.environ.get("AUTOLUCE_SOURCE_URL", manifest.repository)
    ref = os.environ.get("AUTOLUCE_SOURCE_REF", manifest.ref)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not layout.checkout.exists():
        run(["git", "clone", "--no-checkout", "--filter=blob:none", repository, str(layout.checkout)])
    run(["git", "remote", "set-url", "origin", repository], cwd=layout.checkout)
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=layout.checkout)
    result = run(["git", "rev-parse", "FETCH_HEAD"], cwd=layout.checkout, check=True, capture_output=True)
    commit = result.stdout.strip()
    if len(ref) == 40 and commit != ref:
        raise RuntimeError(f"resolved Lucebox commit {commit} does not match manifest pin {ref}")
    layout.pin_file.parent.mkdir(parents=True, exist_ok=True)
    layout.pin_file.write_text(commit + "\n")
    layout.source_record_file.write_text(json.dumps({
        "repository": repository, "requested_ref": ref, "commit": commit, "layout": manifest.layout,
    }, indent=2) + "\n")
    print(f"Pinned Lucebox product: {commit}")
    run(["git", "checkout", "--detach", commit], cwd=layout.checkout)
    backend = detect_backend()
    submodules = layout.submodules(backend) if backend in manifest.supported_backends else []
    if submodules:
        run(
            ["git", "submodule", "update", "--init", "--recursive", "--", *submodules],
            cwd=layout.checkout,
        )
    layout.validate()


def _referenced_manifest_keys() -> set[str]:
    """Manifest keys needed by the selected benchmarks.

    Honors AUTOLUCE_BENCHMARKS (comma list) the same way the harness does; when
    unset, all benchmarks/*.json are considered. A manifest entry with no
    benchmark (e.g. a future model) is not downloaded."""
    selected = os.environ.get("AUTOLUCE_BENCHMARKS")
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
    """Directories scanned for existing GGUFs. AUTOLUCE_MODELS (colon-separated)
    entries are prepended; non-existent dirs are skipped."""
    env = os.environ.get("AUTOLUCE_MODELS")
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
    # with ~1 GB instead of ~35 GB: AUTOLUCE_BENCHMARKS=smoke uv run prepare.py
    dsv4_entry = load_catalog()["deepseek-v4-flash"]
    dsv4_paths = resolve_entry_files(
        dsv4_entry,
        Path(os.environ.get("AUTOLUCE_MODEL_ROOT", str(MODELS_DIR))),
    )
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
        "deepseek-v4-flash": {
            "target": {
                "path": str(dsv4_paths[0]),
                "files": [str(path) for path in dsv4_paths],
                "quant": dsv4_entry.quant,
                "expected_size_bytes": dsv4_entry.expected_size_bytes,
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
            if info.get("path"):
                paths = [Path(value) for value in info.get("files", [info["path"]])]
                if not all(path.is_file() for path in paths):
                    raise FileNotFoundError(
                        f"external model '{name}' is missing: "
                        + ", ".join(str(path) for path in paths if not path.is_file())
                    )
                actual_size = sum(path.stat().st_size for path in paths)
                expected_size = info.get("expected_size_bytes")
                if expected_size is not None and actual_size != expected_size:
                    raise ValueError(f"external model '{name}' size mismatch: {actual_size} != {expected_size}")
                print(f"  Using external {role}: {info['path']}")
                continue
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


def validate_product_backend(layout: SourceLayout, backend: str) -> None:
    """Fail before configuring when host detection chose a non-product backend."""
    if backend not in layout.manifest.supported_backends:
        raise RuntimeError(
            f"Lucebox product backend '{backend}' is unavailable; supported backends are "
            f"{', '.join(layout.manifest.supported_backends)}"
        )


def build_commands(
    layout: SourceLayout, backend: str, jobs: int, *, use_ccache: bool,
) -> tuple[list[str], list[str]]:
    build_dir = layout.build_dir(backend)
    configure = [
        "cmake", "-G", "Ninja",
        "-S", str(layout.cmake_source),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        *layout.cmake_backend_flags(backend),
    ]
    if use_ccache:
        configure += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    build = [
        "cmake", "--build", str(build_dir), "-j", str(jobs), "--target", *layout.build_targets,
    ]
    return configure, build


def build_lucebox() -> None:
    """Configure and build the Lucebox product for the detected accelerator."""
    if shutil.which("cmake") is None:
        print("ERROR: cmake is not installed.", file=sys.stderr)
        sys.exit(1)

    backend = detect_backend()
    layout = SourceLayout.resolve()
    layout.validate()
    layout.require_capability("product-build")
    try:
        validate_product_backend(layout, backend)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    jobs = min(4, max(1, int(os.environ.get("AUTOLUCE_BUILD_JOBS", "4"))))
    cmake_args, build_args = build_commands(layout, backend, jobs, use_ccache=bool(shutil.which("ccache")))
    layout.build_dir(backend).mkdir(parents=True, exist_ok=True)
    print(f"Building for backend: {backend}")

    run(cmake_args)
    run(build_args)
    print(f"Build complete: {layout.build_dir(backend)} (backend={backend})")


def _run_remote_setup(target_name: str, backend: str, provision_tools: bool = False) -> None:
    from autoluce.remote import SSHWorker
    from autoluce.targets import TargetConfig

    target = TargetConfig.load(target_name)
    worker = SSHWorker(target)
    worker.sync_repo()
    worker.ensure_remote_uv()
    root = target.root.rstrip("/")
    if provision_tools:
        packages = ["ccache"]
        worker.run(["sudo", "-n", "apt-get", "update"], lease=True, timeout=1800)
        worker.run(
            ["sudo", "-n", "apt-get", "install", "-y", *packages],
            lease=True,
            timeout=1800,
        )
    backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP"}[backend]
    env_args = [
        "env", "AUTOLUCE_REMOTE_WORKER=1", f"{backend_var}=ON",
        f"AUTOLUCE_MODEL_ROOT={target.model_root or root + '/work/models'}",
        f"AUTOLUCE_BUILD_JOBS={target.build_jobs}",
        f"AUTOLUCE_BUILD_SUBDIR=build-{backend}",
        "AUTOLUCE_BENCHMARKS=deepseek-v4-flash",
    ]
    worker.run(
        [*env_args, f"{root}/.tools/uv", "run", "python", "-m", "autoluce.prepare"],
        lease=True,
        timeout=7200,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare local or remote autoluce worker")
    parser.add_argument("--target")
    parser.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    parser.add_argument("--provision-tools", action="store_true", help="Install missing ccache remotely")
    args = parser.parse_args()
    if args.target and os.environ.get("AUTOLUCE_REMOTE_WORKER") != "1":
        _run_remote_setup(args.target, args.backend, args.provision_tools)
        print(f"Remote setup complete: {args.target} ({args.backend})")
        return
    print("autoluce v2 setup")
    print("=================")
    clone_lucebox()
    download_models()
    build_lucebox()
    print("\nSetup complete. Run `uv run autoluce source status` next.")


if __name__ == "__main__":
    main()
