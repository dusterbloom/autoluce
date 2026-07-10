"""Remember the local agent identity after one-time registration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_AGENT_CONFIG = Path("~/.config/autoggml/agent.json").expanduser()


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    name: str
    schema_version: int = 1

    @classmethod
    def path(cls) -> Path:
        return Path(os.environ.get("AUTOGGML_AGENT_CONFIG", DEFAULT_AGENT_CONFIG)).expanduser()

    @classmethod
    def load(cls) -> "AgentIdentity | None":
        path = cls.path()
        return cls(**json.loads(path.read_text())) if path.exists() else None

    def write(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        path.chmod(0o600)
