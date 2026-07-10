"""Reusable process-safe JSON persistence for small coordinator state."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class AtomicJsonStore:
    def __init__(self, path: Path, default_factory: Callable[[], dict]) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.default_factory = default_factory

    def update(self, operation: Callable[[dict], T]) -> T:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else self.default_factory()
            result = operation(state)
            fd, temporary = tempfile.mkstemp(prefix=self.path.stem + "-", suffix=".json", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(state, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return result
