"""Single policy gate for human- or agent-authored engine patches."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


APPROVED_PREFIXES = (
    "server/src/",
    "server/include/",
    "server/deps/llama.cpp/ggml/",
)
APPROVED_FILES = {"server/CMakeLists.txt"}


class CandidatePatchGate:
    _header = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)

    def validate(self, patch: bytes) -> list[str]:
        try:
            text = patch.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("candidate patch must be UTF-8 text") from error
        paths = []
        for source, destination in self._header.findall(text):
            for value in (source, destination):
                path = PurePosixPath(value)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe patch path: {value}")
                if not value.startswith(APPROVED_PREFIXES) and value not in APPROVED_FILES:
                    raise ValueError(f"not an approved Lucebox product path: {value}")
            paths.append(destination)
        if not paths:
            raise ValueError("candidate patch has no git diff headers")
        return sorted(set(paths))
