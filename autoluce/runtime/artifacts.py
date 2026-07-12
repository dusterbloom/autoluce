"""Content identity for an executable and its resolved shared libraries."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class ArtifactFingerprint:
    name: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuntimeArtifactClosure:
    executable: ArtifactFingerprint
    dependencies: tuple[ArtifactFingerprint, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(name: str, path: Path) -> ArtifactFingerprint:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"runtime artifact is not a regular file: {resolved}")
    return ArtifactFingerprint(name, str(resolved), _sha256(resolved), resolved.stat().st_size)


def _parse_ldd(output: str) -> list[tuple[str, Path]]:
    dependencies: list[tuple[str, Path]] = []
    missing: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"linux-vdso\.so\.\d+\s+\(0x[0-9a-fA-F]+\)", line):
            continue
        missing_match = re.fullmatch(r"(\S+)\s+=>\s+not found", line)
        if missing_match:
            missing.append(missing_match.group(1))
            continue
        mapped = re.fullmatch(r"(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)", line)
        if mapped:
            dependencies.append((mapped.group(1), Path(mapped.group(2))))
            continue
        loader = re.fullmatch(r"(/\S+)\s+\(0x[0-9a-fA-F]+\)", line)
        if loader:
            path = Path(loader.group(1))
            dependencies.append((path.name, path))
            continue
        raise RuntimeError(f"unrecognized ldd output: {line}")
    if missing:
        raise RuntimeError("missing runtime dependencies: " + ", ".join(sorted(missing)))
    return dependencies


def capture_runtime_artifact_closure(
    executable: Path,
    *,
    env: Mapping[str, str],
    runner: Callable = subprocess.run,
) -> RuntimeArtifactClosure:
    """Resolve and hash the exact ELF artifacts used to launch the product."""
    process = runner(
        ["ldd", str(executable)],
        capture_output=True,
        text=True,
        check=False,
        env=dict(env),
    )
    if process.returncode:
        detail = "\n".join(part for part in (process.stdout, process.stderr) if part).strip()
        raise RuntimeError(detail or f"ldd failed for {executable}")
    dependencies = tuple(
        sorted(
            (_fingerprint(name, path) for name, path in _parse_ldd(process.stdout)),
            key=lambda item: (item.name, item.path),
        )
    )
    return RuntimeArtifactClosure(
        executable=_fingerprint(executable.name, executable),
        dependencies=dependencies,
    )


def require_stable_runtime_artifacts(
    built: RuntimeArtifactClosure,
    measured: RuntimeArtifactClosure,
) -> None:
    if built == measured:
        return
    before = {item.path: item.sha256 for item in (built.executable, *built.dependencies)}
    after = {item.path: item.sha256 for item in (measured.executable, *measured.dependencies)}
    changed = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    raise RuntimeError("runtime artifacts changed after the build: " + ", ".join(changed))
