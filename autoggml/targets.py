"""Target configuration for local and SSH-backed research workers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGETS_PATH = Path("~/.config/autoggml/targets.toml").expanduser()


@dataclass(frozen=True)
class TargetConfig:
    name: str
    transport: str
    host: str | None = None
    root: str | None = None
    model_root: str | None = None
    lock_path: str = "/tmp/autoggml-gpu.lock"
    build_jobs: int = 4

    @classmethod
    def load(cls, name: str, path: Path | None = None) -> "TargetConfig":
        path = Path(os.environ.get("AUTOGGML_TARGETS_FILE", path or DEFAULT_TARGETS_PATH)).expanduser()
        raw: dict = {}
        if path.exists():
            raw = tomllib.loads(path.read_text()).get("targets", {}).get(name, {})

        prefix = "AUTOGGML_TARGET_"
        host = os.environ.get(prefix + "HOST", raw.get("host"))
        root = os.environ.get(prefix + "ROOT", raw.get("root"))
        model_root = os.environ.get(prefix + "MODEL_ROOT", raw.get("model_root"))
        transport = os.environ.get(prefix + "TRANSPORT", raw.get("transport", "ssh" if host else "local"))
        lock_path = os.environ.get(prefix + "LOCK_PATH", raw.get("lock_path", "/tmp/autoggml-gpu.lock"))
        build_jobs = int(os.environ.get(prefix + "BUILD_JOBS", raw.get("build_jobs", 4)))

        if transport not in {"local", "ssh"}:
            raise ValueError(f"target '{name}' has unsupported transport '{transport}'")
        if transport == "ssh" and (not host or not root):
            raise ValueError(f"target '{name}' requires host and root")
        if build_jobs < 1 or build_jobs > 4:
            raise ValueError("build_jobs must be between 1 and 4")
        return cls(name, transport, host, root, model_root, lock_path, build_jobs)
