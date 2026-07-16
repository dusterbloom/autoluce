"""Repository model catalog with external-file and GGUF-shard support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autoluce import ROOT
from autoluce.profiles import ModelProfile


CATALOG_PATH = ROOT / "models" / "catalog.yaml"


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    quant: str
    files: list[str]
    repository: str | None = None
    revision: str | None = None
    sha256: str | None = None
    expected_size_bytes: int | None = None
    path_env: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def first_file(self) -> str:
        if not self.files:
            raise ValueError(f"model '{self.model_id}' has no files")
        return self.files[0]


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, ModelEntry]:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    entries: dict[str, ModelEntry] = {}
    for model_id, value in (raw.get("models", {}) or {}).items():
        files = value.get("files") or ([value["file"]] if value.get("file") else [])
        entries[model_id] = ModelEntry(
            model_id=model_id,
            quant=value.get("quant", "unknown"),
            files=list(files),
            repository=value.get("repository"),
            revision=value.get("revision"),
            sha256=value.get("sha256"),
            expected_size_bytes=value.get("expected_size_bytes"),
            path_env=value.get("path_env"),
            metadata=value.get("metadata", {}),
        )
    return entries


def resolve_entry_files(entry: ModelEntry, model_root: Path | None = None) -> list[Path]:
    override = os.environ.get(entry.path_env, "") if entry.path_env else ""
    if override:
        first = Path(override).expanduser()
        if len(entry.files) == 1:
            return [first]
        root = first.parent
        return [first, *(root / name for name in entry.files[1:])]
    root = Path(model_root or os.environ.get("AUTOLUCE_MODEL_ROOT", ROOT / "work" / "models")).expanduser()
    return [Path(name) if Path(name).is_absolute() else root / name for name in entry.files]


def profile_model(entry: ModelEntry, paths: list[Path], sha256: str | None = None) -> ModelProfile:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {', '.join(missing)}")
    size = sum(path.stat().st_size for path in paths)
    if entry.expected_size_bytes is not None and size != entry.expected_size_bytes:
        raise ValueError(
            f"model '{entry.model_id}' size mismatch: expected {entry.expected_size_bytes}, got {size}"
        )
    return ModelProfile(entry.model_id, entry.quant, [str(path) for path in paths], size, sha256, entry.metadata or {})
