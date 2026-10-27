"""Small dependency-free loader for local environment settings."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE settings without overriding real environment values."""
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
