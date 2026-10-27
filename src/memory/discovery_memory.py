"""Small persistent memory layer for prediction and external feedback records."""

from __future__ import annotations

import json
import os
from typing import Any


class DiscoveryMemory:
    def __init__(self, path: str) -> None:
        self.path = path

    def append_feedback(self, record: dict[str, Any]) -> None:
        records = self.load_feedback()
        records.append(record)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(records[-2000:], handle, indent=2)

    def load_feedback(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
