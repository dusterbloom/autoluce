"""Persistent client connection settings for the shared coordinator."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_TEAM_CONFIG = Path("~/.config/autoggml/team.json").expanduser()


@dataclass(frozen=True)
class TeamConnection:
    url: str
    token: str
    schema_version: int = 1

    @classmethod
    def path(cls) -> Path:
        return Path(os.environ.get("AUTOGGML_TEAM_CONFIG", DEFAULT_TEAM_CONFIG)).expanduser()

    @classmethod
    def load(cls) -> "TeamConnection | None":
        url = os.environ.get("AUTOGGML_COORDINATOR_URL")
        token = os.environ.get("AUTOGGML_COORDINATOR_TOKEN")
        if url:
            if not token:
                raise ValueError("AUTOGGML_COORDINATOR_TOKEN is required for a shared coordinator")
            return cls(url, token)
        path = cls.path()
        return cls(**json.loads(path.read_text())) if path.exists() else None

    def write(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("team URL must start with http:// or https://")
        if not self.token:
            raise ValueError("team token must not be empty")
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        path.chmod(0o600)
