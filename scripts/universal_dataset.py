"""
Universal Materials Dataset Generator.
Creates a curated dataset of polymers, carbon allotropes, structural metals,
semiconductors, and perovskites for universal MCRSA_EGNN fine-tuning.
"""

from __future__ import annotations
import json
import os
import math
import random
import numpy as np
from typing import Dict, Any, List


DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)
UNIVERSAL_DATASET_PATH = os.path.join(DATA_DIR, "universal_materials_5k.json")


def generate_curated_universal_dataset() -> List[Dict[str, Any]]:
    """Builds a curated, multi-category dataset of 3,200 materials."""
    dataset = []

    # 1. POLYMERS & MACROMOLECULES (Combs, Bottles, Packaging, Bio-nanocomposites)
    polymer_templates = [
        {
            "formula": "(C3H6)n_PP",
            "name": "Isotactic Polypropylene",
            "category": "Polymer",
            "elements": ["C", "C", "C", "H", "H", "H", "H", "H", "H"],
            "target_energy": -0.42,
            "coords": [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.10, 1.35, 0.0],
                       [0.0, 1.09, 0.0], [1.54, -1.09, 0.0], [2.60, 1.50, 0.89],
                       [-0.50, -0.50, 0.89], [1.20, 0.50, 0.89], [3.10, 1.30, -0.89]]
        },
        {
            "formula": "C6H10O5_Cellulose",
            "name": "Crystalline Cellulose Backbone",
            "category": "Bio-Polymer",
            "elements": ["C", "C", "C", "C", "C", "C", "O", "O", "O", "O", "O", "H", "H", "H", "H"],
            "target_energy": -1.85,
            "coords": [[-2.0, -0.8, 0.0], [-1.2, 0.4, 0.2], [0.2, 0.4, -0.2], [0.9, -0.8, -0.1],
                       [0.0, -1.9, 0.1], [-1.3, -1.9, -0.1], [2.2, -0.7, -0.3], [-2.7, 0.3, 0.4],
                       [0.2, 1.6, 0.1], [-1.5, 1.4, 0.5], [1.2, -2.5, 0.3], [-1.2, 0.9, 1.1],
                       [0.3, 0.9, -1.1], [0.0, -2.4, 1.0], [-2.9, -0.5, 0.1]]
        },
        {
            "formula": "(C10H8O4)n_PET",
            "name": "Polyethylene Terephthalate",
            "category": "Polymer",
            "elements": ["C", "C", "C", "C", "C", "C", "C", "C", "O", "O", "O", "O", "H", "H", "H", "H"],
            "target_energy": -0.68,
            "coords": [[-1.4, 0.0, 0.0], [-0.7, 1.2, 0.0], [0.7, 1.2, 0.0], [1.4, 0.0, 0.0],
                       [0.7, -1.2, 0.0], [-0.7, -1.2, 0.0], [2.8, 0.0, 0.0], [-2.8, 0.0, 0.0],
                       [3.4, 1.1, 0.0], [3.4, -1.1, 0.0], [-3.4, 1.1, 0.0], [-3.4, -1.1, 0.0],
                       [-1.2, 2.1, 0.0], [1.2, 2.1, 0.0], [1.2, -2.1, 0.0], [-1.2, -2.1, 0.0]]
        },
        {
            "formula": "Bio_Nanocomposite_Engineered",
            "name": "Cross-Linked Bio-Nanocomposite Superlattice",
            "category": "AI-Redesigned Material",
            "elements": ["C", "C", "C", "C", "C", "O", "O", "O", "C", "O", "C", "C", "O", "H", "H", "H", "H"],
            "target_energy": -2.18,
            "coords": [[-2.0, -0.8, 0.0], [-1.2, 0.4, 0.2], [0.2, 0.4, -0.2], [0.9, -0.8, -0.1],
                       [0.0, -1.9, 0.1], [-1.3, -1.9, -0.1], [2.2, -0.7, -0.3], [-2.7, 0.3, 0.4],
                       [2.8, 0.4, 0.1], [2.3, 1.5, 0.4], [4.3, 0.3, 0.0], [5.0, 1.5, -0.3],
                       [4.8, 2.7, 0.2], [-1.2, 1.0, 1.1], [0.3, 1.0, -1.1], [0.0, -2.5, 1.0], [4.5, -0.2, -0.9]]
        }
    ]

    # Generate conformational variants of polymers
    for idx, poly in enumerate(polymer_templates):
        for var in range(250):
            noise = (np.random.rand(len(poly["elements"]), 3) - 0.5) * 0.08
            perturbed_coords = (np.array(poly["coords"]) + noise).tolist()
            energy_noise = (random.random() - 0.5) * 0.05
            dataset.append({
                "material_id": f"poly_{idx}_{var}",
                "formula": f"{poly['formula']}_v{var}",
                "elements": poly["elements"],
                "coordinates_3d": [[round(c, 3) for c in pt] for pt in perturbed_coords],
                "target_energy": round(poly["target_energy"] + energy_noise, 3),
                "category": poly["category"],
            })

    # 2. CARBON ALLOTROPES (Graphite, Graphene, Diamond)
    carbon_allotropes = [
        {
            "formula": "C_Graphite",
            "elements": ["C"] * 8,
            "target_energy": 0.00,
            "coords": [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [2.13, 1.23, 0.0], [1.42, 2.46, 0.0],
                       [0.0, 0.0, 3.35], [1.42, 0.0, 3.35], [2.13, 1.23, 3.35], [1.42, 2.46, 3.35]]
        },
        {
            "formula": "C_Diamond",
            "elements": ["C"] * 8,
            "target_energy": -0.14,
            "coords": [[0.0, 0.0, 0.0], [1.78, 1.78, 0.0], [0.0, 1.78, 1.78], [1.78, 0.0, 1.78],
                       [0.89, 0.89, 0.89], [2.67, 2.67, 0.89], [0.89, 2.67, 2.67], [2.67, 0.89, 2.67]]
        }
    ]
    for idx, carb in enumerate(carbon_allotropes):
        for var in range(200):
            noise = (np.random.rand(len(carb["elements"]), 3) - 0.5) * 0.06
            perturbed_coords = (np.array(carb["coords"]) + noise).tolist()
            dataset.append({
                "material_id": f"carbon_{idx}_{var}",
                "formula": f"{carb['formula']}_v{var}",
                "elements": carb["elements"],
                "coordinates_3d": [[round(c, 3) for c in pt] for pt in perturbed_coords],
                "target_energy": round(carb["target_energy"] + (random.random() - 0.5) * 0.03, 3),
                "category": "Carbon Allotrope",
            })

    # 3. STRUCTURAL METALS (Aluminum, Iron Steel, Copper, Titanium)
    metals = [
        {"formula": "Al_FCC", "el": "Al", "target_energy": -0.22, "a": 4.05},
        {"formula": "Fe_BCC", "el": "Fe", "target_energy": -0.85, "a": 2.87},
        {"formula": "Cu_FCC", "el": "Cu", "target_energy": -0.38, "a": 3.61},
        {"formula": "Ti_HCP", "el": "Ti", "target_energy": -1.25, "a": 2.95},
    ]
    for met in metals:
        a = met["a"]
        base_fcc = [
            [0.0, 0.0, 0.0], [a/2, a/2, 0.0], [a/2, 0.0, a/2], [0.0, a/2, a/2],
            [a, a, 0.0], [a, 0.0, a], [0.0, a, a], [a, a, a]
        ]
        for var in range(200):
            noise = (np.random.rand(8, 3) - 0.5) * 0.05
            perturbed_coords = (np.array(base_fcc) + noise).tolist()
            dataset.append({
                "material_id": f"metal_{met['el']}_{var}",
                "formula": f"{met['formula']}_v{var}",
                "elements": [met["el"]] * 8,
                "coordinates_3d": [[round(c, 3) for c in pt] for pt in perturbed_coords],
                "target_energy": round(met["target_energy"] + (random.random() - 0.5) * 0.04, 3),
                "category": "Metal & Structural Alloy",
            })

    # 4. CERAMICS, SILICA & MINERAL PEROVSKITES
    perovskites = [
        {"formula": "SrTiO3", "A": "Sr", "B": "Ti", "X": "O", "energy": -3.65},
        {"formula": "BaTiO3", "A": "Ba", "B": "Ti", "X": "O", "energy": -3.42},
        {"formula": "CaTiO3", "A": "Ca", "B": "Ti", "X": "O", "energy": -3.55},
        {"formula": "LaFeO3", "A": "La", "B": "Fe", "X": "O", "energy": -3.20},
        {"formula": "SiO2_Quartz", "A": "Si", "B": "O", "X": "O", "energy": -2.85},
    ]
    for p in perovskites:
        pts = [
            [0.0, 0.0, 0.0], [2.0, 2.0, 2.0],
            [2.0, 2.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0]
        ]
        els = [p["A"], p["B"], p["X"], p["X"], p["X"]]
        for var in range(200):
            noise = (np.random.rand(5, 3) - 0.5) * 0.05
            perturbed_coords = (np.array(pts) + noise).tolist()
            dataset.append({
                "material_id": f"ceramic_{p['formula']}_{var}",
                "formula": f"{p['formula']}_v{var}",
                "elements": els,
                "coordinates_3d": [[round(c, 3) for c in pt] for pt in perturbed_coords],
                "target_energy": round(p["energy"] + (random.random() - 0.5) * 0.05, 3),
                "category": "Ceramic & Perovskite",
            })

    random.shuffle(dataset)
    with open(UNIVERSAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated universal dataset: {len(dataset)} records -> {UNIVERSAL_DATASET_PATH}")
    return dataset


if __name__ == "__main__":
    generate_curated_universal_dataset()
