"""
Dynamic Element Registry for Causal MCRSA Architecture.
Discovers and registers chemical elements dynamically row-by-row on the fly.
Zero static dictionary lookup. Expands vocabulary autonomously as novel elements appear.
"""

import json
from typing import List, Dict, Any
import torch


class DynamicElementRegistry:
    """
    Dynamically registers chemical elements as they appear in data rows.
    Zero static dictionary lookup. Expands vocabulary autonomously.
    """
    def __init__(self):
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: List[str] = []

    def get_or_add(self, symbol: str) -> int:
        if symbol not in self.symbol_to_id:
            new_id = len(self.id_to_symbol)
            self.symbol_to_id[symbol] = new_id
            self.id_to_symbol.append(symbol)
        return self.symbol_to_id[symbol]

    def encode_elements(self, elements: List[str]) -> torch.Tensor:
        return torch.tensor([self.get_or_add(e) for e in elements], dtype=torch.long)

    @property
    def total_elements(self) -> int:
        return len(self.id_to_symbol)

    def save(self, file_path: str) -> None:
        """Saves registry vocabulary to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbol_to_id": self.symbol_to_id,
                "id_to_symbol": self.id_to_symbol
            }, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "DynamicElementRegistry":
        """Loads registry vocabulary from JSON file."""
        registry = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            registry.symbol_to_id = data["symbol_to_id"]
            registry.id_to_symbol = data["id_to_symbol"]
        return registry

