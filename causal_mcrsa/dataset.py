"""
Dataset loader and target calibrator for Causal MCRSA Architecture.
Zero data leakage: Computes target normalization (mean and std) strictly
across the training partition (85%, N=16,088).
Provides deterministic 85/15 train/validation partitions.
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np


class PerovskiteDataset:
    """
    Handles perovskite dataset ingestion and Z-score target calibration.
    Strictly calculates mean and std on the training partition to avoid data leakage.
    """
    def __init__(self, data_path: str = None, train_ratio: float = 0.85, seed: int = 42):
        if data_path is None:
            candidates = [
                os.path.join("Data", "perovskites_3d_dataset.json"),
                os.path.join("..", "Data", "perovskites_3d_dataset.json"),
            ]
            data_path = next((p for p in candidates if os.path.exists(p)), None)
            if data_path is None:
                raise FileNotFoundError("perovskites_3d_dataset.json not found in Data directory.")

        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data: List[Dict[str, Any]] = json.load(f)

        # Deterministic 85/15 train/validation partition
        shuffled = list(self.raw_data)
        random.Random(seed).shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)

        self.train_set = shuffled[:split_idx]
        self.val_set = shuffled[split_idx:]

        # Zero Data Leakage: Calculate target normalization strictly on training partition
        train_targets = np.array([float(d["target_energy"]) for d in self.train_set], dtype=np.float32)
        self.target_mean = float(np.mean(train_targets))
        self.target_std = max(float(np.std(train_targets)), 1e-6)

    def get_splits(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Returns (train_set, val_set)."""
        return self.train_set, self.val_set
