"""
Patch toolkit for explicit Lucebox product or vendored-ggml roots.

Each patch function accepts the exact source scope it mutates and
mutates one or more source files. Patches are reversible with git.

Intended use from experiment.py:

    from autoggml.loop.patches import apply_march_native, apply_speculative_candidates
    from pathlib import Path

    from autoggml.source_layout import SourceLayout
    layout = SourceLayout.resolve()
    repo = layout.cmake_source       # product CMake/source helpers
    vendor = layout.vendor_root      # upstream-compatible ggml helpers
    patches.apply_march_native(repo)
    patches.apply_speculative_candidates(repo, n_draft=8)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

PatchFn = Callable[[Path], dict[str, str]]


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text()


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def _git_checkout(repo: Path, rel_path: str) -> None:
    subprocess.run(
        ["git", "checkout", "--", rel_path],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def revert_all(repo: Path, files: list[str]) -> None:
    """Revert all patched files to the pinned commit."""
    for rel in files:
        _git_checkout(repo, rel)


CMAKE_PROJECT_ANCHORS = ("project(dflash LANGUAGES", "project(llama.cpp C CXX)")


def _anchor_insert(text: str, anchor: str, insertion: str) -> str | None:
    """Insert `insertion` after the first occurrence of `anchor`, or return None if not found."""
    if anchor not in text:
        return None
    return text.replace(anchor, f"{anchor}\n{insertion}", 1)


def apply_march_native(repo: Path, flags: str | None = None) -> dict[str, str]:
    """
    Append native CPU architecture flags to the main CMakeLists.txt.
    Default: -march=native -mtune=native when compiler is GCC/Clang.
    """
    path = repo / "CMakeLists.txt"
    text = _read(path)

    flag_str = flags or "-march=native -mtune=native"
    marker = "# autoggml: march-native"

    if marker in text:
        return {"patch": "march_native", "status": "already_applied", "flags": flag_str}

    insertion = f"""
{marker}
if ((CMAKE_C_COMPILER_ID MATCHES "GNU|Clang") AND NOT EMSCRIPTEN)
    add_compile_options({flag_str})
endif()
"""
    anchor = next((candidate for candidate in CMAKE_PROJECT_ANCHORS if candidate in text), None)
    new_text = _anchor_insert(text, anchor, insertion) if anchor else None
    if new_text is None:
        return {"patch": "march_native", "status": "no_anchor", "flags": flag_str}

    _write(path, new_text)
    return {"patch": "march_native", "status": "applied", "flags": flag_str}


def apply_lto(repo: Path) -> dict[str, str]:
    """Enable link-time optimization (LTO) in CMake."""
    path = repo / "CMakeLists.txt"
    text = _read(path)
    marker = "# autoggml: lto"

    if marker in text:
        return {"patch": "lto", "status": "already_applied"}

    insertion = f"""
{marker}
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
"""
    anchor = next((candidate for candidate in CMAKE_PROJECT_ANCHORS if candidate in text), None)
    new_text = _anchor_insert(text, anchor, insertion) if anchor else None
    if new_text is None:
        return {"patch": "lto", "status": "no_anchor"}

    _write(path, new_text)
    return {"patch": "lto", "status": "applied"}


def apply_speculative_candidates(repo: Path, n_draft: int = 8) -> dict[str, str]:
    """
    Bump the default number of draft tokens for speculative decoding.
    Tries common locations in common/speculative.cpp and llama.cpp.
    """
    changed: list[str] = []
    files = [repo / "common" / "speculative.cpp", repo / "src" / "llama.cpp"]
    total_replacements = 0

    for path in files:
        if not path.exists():
            continue
        text = path.read_text()
        # Match simple assignments like `int n_draft = 4;` or `.n_draft = 4;`
        pattern = re.compile(r"(n_draft\s*=\s*)(\d+)")
        new_text, count = pattern.subn(lambda m: f"{m.group(1)}{n_draft}", text, count=3)
        if count:
            _write(path, new_text)
            changed.append(str(path.relative_to(repo)))
            total_replacements += count

    return {
        "patch": "speculative_candidates",
        "status": "applied" if changed else "no_match",
        "n_draft": str(n_draft),
        "files": ",".join(changed),
        "replacements": str(total_replacements),
    }


def apply_graph_threads(repo: Path, n_threads: int = 4) -> dict[str, str]:
    """
    Change the default number of threads used by the GGML graph scheduler.
    Looks for common default initializations in ggml/src/ggml.c or similar.
    """
    # Limit scope to the GGML source tree.
    paths = [
        p for p in (list((repo / "ggml").rglob("ggml*.c")) + list((repo / "ggml").rglob("ggml*.cpp")))
        if p.is_file()
    ]
    changed: list[str] = []
    total_replacements = 0

    for path in paths:
        text = path.read_text()
        # Only touch simple default assignments, not loop variables or function parameters.
        pattern = re.compile(r"^\s*(int\s+)?n_threads\s*=\s*\d+\s*;", re.MULTILINE)
        new_text, count = pattern.subn(lambda m: re.sub(r"\d+", str(n_threads), m.group(0), count=1), text, count=3)
        if count:
            _write(path, new_text)
            changed.append(str(path.relative_to(repo)))
            total_replacements += count

    return {
        "patch": "graph_threads",
        "status": "applied" if changed else "no_match",
        "n_threads": str(n_threads),
        "files": ",".join(changed),
        "replacements": str(total_replacements),
    }


def apply_kv_cache_type(repo: Path, k_type: str = "f16", v_type: str = "f16") -> dict[str, str]:
    """
    Set default KV cache data types in llama.cpp. Common values: f32, f16, q8_0, q4_0.
    """
    path = repo / "src" / "llama.cpp"
    if not path.exists():
        return {"patch": "kv_cache_type", "status": "no_target"}

    text = path.read_text()
    original = text

    # Try to find default type enum assignments (best-effort)
    text = re.sub(
        r"(cache_type_k\s*=\s*LLAMA_KV_CACHE_TYPE_)\w+",
        lambda m: f"{m.group(1)}{k_type.upper()}",
        text,
    )
    text = re.sub(
        r"(cache_type_v\s*=\s*LLAMA_KV_CACHE_TYPE_)\w+",
        lambda m: f"{m.group(1)}{v_type.upper()}",
        text,
    )

    if text != original:
        _write(path, text)
        return {
            "patch": "kv_cache_type",
            "status": "applied",
            "k_type": k_type,
            "v_type": v_type,
        }

    return {"patch": "kv_cache_type", "status": "no_match", "k_type": k_type, "v_type": v_type}


# Registry for programmatic use.
PATCH_REGISTRY: dict[str, PatchFn] = {
    "march_native": apply_march_native,
    "lto": apply_lto,
    "speculative_candidates": apply_speculative_candidates,
    "graph_threads": apply_graph_threads,
    "kv_cache_type": apply_kv_cache_type,
}


def apply_patch(repo: Path, name: str, **kwargs) -> dict[str, str]:
    """Apply a named patch with keyword arguments."""
    if name not in PATCH_REGISTRY:
        raise ValueError(f"Unknown patch: {name}. Available: {list(PATCH_REGISTRY)}")
    return PATCH_REGISTRY[name](repo, **kwargs)
