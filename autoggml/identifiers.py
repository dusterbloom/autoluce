"""Content-derived identifiers shared across coordination domains."""

from __future__ import annotations

import hashlib


def stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join(values).encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"
