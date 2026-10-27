"""Physics-informed external features for perovskite graph inputs."""

from __future__ import annotations

from typing import Sequence

import torch

from src.physics_constants import ELEMENT_PHYSICS


_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn "
    "Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(_ELEMENT_SYMBOLS, 1)}

# Group and period are stable, useful descriptors even when a material is not in
# the local training table. Values are curated from the periodic table.
_GROUPS = {
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Fr": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2,
    "B": 13, "Al": 13, "Ga": 13, "In": 13, "Tl": 13, "Nh": 13,
    "C": 14, "Si": 14, "Ge": 14, "Sn": 14, "Pb": 14, "Fl": 14,
    "N": 15, "P": 15, "As": 15, "Sb": 15, "Bi": 15, "Mc": 15,
    "O": 16, "S": 16, "Se": 16, "Te": 16, "Po": 16, "Lv": 16,
    "F": 17, "Cl": 17, "Br": 17, "I": 17, "At": 17, "Ts": 17,
    "He": 18, "Ne": 18, "Ar": 18, "Kr": 18, "Xe": 18, "Rn": 18, "Og": 18,
}
_PERIODS = {}
for _symbol, _atomic_number in _ATOMIC_NUMBERS.items():
    _PERIODS[_symbol] = (
        1 if _atomic_number <= 2 else
        2 if _atomic_number <= 10 else
        3 if _atomic_number <= 18 else
        4 if _atomic_number <= 36 else
        5 if _atomic_number <= 54 else
        6 if _atomic_number <= 86 else 7
    )
for _symbol, _group in {
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8,
    "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12, "La": 3, "Hf": 4, "Ta": 5, "W": 6,
    "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12, "Ac": 3, "Rf": 4,
    "Db": 5, "Sg": 6, "Bh": 7, "Hs": 8, "Mt": 9, "Ds": 10, "Rg": 11, "Cn": 12,
}.items():
    _GROUPS[_symbol] = _group


class MaterialFeaturePipeline:
    """Build node and global features from composition and 3D geometry."""

    node_feature_names = (
        "ionic_radius", "electronegativity", "valence", "atomic_mass",
        "atomic_number", "period", "group", "radius_mass_ratio",
    )
    global_feature_names = (
        "mean_pair_distance", "std_pair_distance", "cell_scale",
        "composition_entropy", "cation_anion_radius_ratio", "mean_atomic_number",
    )

    @property
    def node_feature_dim(self) -> int:
        return len(self.node_feature_names)

    @property
    def global_feature_dim(self) -> int:
        return len(self.global_feature_names)

    def _properties(self, element: str) -> dict[str, float]:
        physics = ELEMENT_PHYSICS.get(
            element,
            {"radius": 1.0, "electronegativity": 1.5, "valence": 0.0, "mass": 50.0},
        )
        atomic_number = float(_ATOMIC_NUMBERS.get(element, 0))
        period = float(_PERIODS.get(element, 0))
        group = float(_GROUPS.get(element, 0))
        radius = float(physics["radius"])
        mass = float(physics["mass"])
        return {
            "ionic_radius": radius,
            "radius": radius,
            "electronegativity": float(physics["electronegativity"]),
            "valence": float(physics["valence"]),
            "atomic_mass": mass / 100.0,
            "atomic_number": atomic_number,
            "period": period,
            "group": group,
            "radius_mass_ratio": radius / max(mass, 1.0),
        }

    def node_features(self, elements: Sequence[str]) -> torch.Tensor:
        rows = []
        for element in elements:
            p = self._properties(element)
            rows.append([p[name] for name in self.node_feature_names])
        return torch.tensor(rows, dtype=torch.float32)

    def global_features(
        self, elements: Sequence[str], coordinates: Sequence[Sequence[float]]
    ) -> torch.Tensor:
        coords = torch.tensor(coordinates, dtype=torch.float32)
        if len(coords) > 1:
            distances = torch.pdist(coords)
            mean_distance = float(distances.mean())
            std_distance = float(distances.std(unbiased=False))
        else:
            mean_distance = std_distance = 0.0
        cell_scale = float(coords.std(unbiased=False)) if coords.numel() else 0.0
        properties = [self._properties(element) for element in elements]
        radii = [p["radius"] for p in properties]
        weights = torch.tensor(radii, dtype=torch.float32)
        probabilities = weights / weights.sum().clamp_min(1e-6)
        entropy = float(-(probabilities * torch.log2(probabilities.clamp_min(1e-6))).sum())
        cation_radius = float(sum(radii[:2]) / 2.0) if len(radii) >= 2 else 1.0
        anion_radius = float(sum(radii[2:]) / max(len(radii[2:]), 1)) if len(radii) > 2 else 1.0
        return torch.tensor(
            [
                mean_distance,
                std_distance,
                cell_scale,
                entropy,
                cation_radius / max(anion_radius, 1e-6),
                sum(p["atomic_number"] for p in properties) / max(len(properties), 1),
            ],
            dtype=torch.float32,
        )

    def tensors(
        self, elements: Sequence[str], coordinates: Sequence[Sequence[float]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.node_features(elements).to(device),
            self.global_features(elements, coordinates).to(device),
        )

    def build_features(
        self, elements: Sequence[str], coordinates: Sequence[Sequence[float]] | None = None
    ) -> torch.Tensor:
        """Alias for node_features, accepting optional coordinates."""
        return self.node_features(elements)

