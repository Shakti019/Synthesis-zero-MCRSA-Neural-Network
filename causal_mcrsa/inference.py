"""
Production Inference Engine for Causal MCRSA Architecture.
Provides single-structure inference with decomposed chemical enthalpy & non-negative steric strain,
calibrated uncertainty, dual-unit reporting (eV/cell and eV/atom), Periodic Boundary Conditions,
and high-throughput screening across composition spaces.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn

from .registry import DynamicElementRegistry
from .model import Causal_MCRSA_EGNN


class CausalMCRSAPredictor:
    """
    High-level predictor for Causal MCRSA Architecture.
    Decomposes formation energy into:
    Net Energy ΔEf = E_chemical + E_strain, with E_strain >= 0 strictly guaranteed.
    """
    def __init__(
        self,
        model: Causal_MCRSA_EGNN,
        registry: DynamicElementRegistry,
        target_mean: float = 1.4682,
        target_std: float = 0.7408,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.registry = registry
        self.target_mean = target_mean
        self.target_std = target_std

    @torch.no_grad()
    def predict(self, elements: List[str], coords: List[List[float]]) -> Dict[str, Any]:
        """
        Runs calibrated causal inference on a single crystal structure.
        Returns dual units (eV/cell and equivalent eV/atom).
        """
        elem_ids = self.registry.encode_elements(elements).to(self.device)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)

        out = self.model(elem_ids, coords_tensor)

        norm_e_chem = out["chemical_energy"].item()
        norm_e_strain = out["steric_strain"].item()
        norm_Ef = out["net_energy_Ef"].item()

        # Decompose physical energies
        e_chem = norm_e_chem * self.target_std + self.target_mean
        e_strain = norm_e_strain * self.target_std  # Strictly >= 0
        pred_Ef = e_chem + e_strain                 # Exact total predicted ΔEf (eV/cell)
        uncertainty = out["uncertainty"].item() * self.target_std

        # Thermodynamic classification (Ef <= 0.20 eV/cell on/near hull)
        is_stable = pred_Ef <= 0.20
        is_metastable = pred_Ef <= 0.50

        if is_stable:
            classification = "Thermodynamically Stable (On/Near Hull)"
        elif is_metastable:
            classification = "Metastable Synthesizable Phase"
        else:
            classification = "Thermodynamically Unstable Phase"

        return {
            "formula": "".join(elements),
            "elements": elements,
            "predicted_energy_Ef": pred_Ef,
            "predicted_energy_atom": pred_Ef / 5.0,
            "chemical_enthalpy": e_chem,
            "chemical_enthalpy_atom": e_chem / 5.0,
            "steric_strain_penalty": e_strain,
            "steric_strain_penalty_atom": e_strain / 5.0,
            "uncertainty_sigma": uncertainty,
            "uncertainty_sigma_atom": uncertainty / 5.0,
            "is_stable": is_stable,
            "is_metastable": is_metastable,
            "classification": classification,
            "units": "eV/unit_cell (5-atom) and eV/atom"
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Saves model weights, registry, and normalization constants."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "units": "eV/unit_cell",
            "registry_symbol_to_id": self.registry.symbol_to_id,
            "registry_id_to_symbol": self.registry.id_to_symbol,
            "use_pbc": getattr(self.model, "use_pbc", True),
            "use_3body": getattr(self.model, "use_3body", True)
        }, checkpoint_path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: torch.device = None) -> "CausalMCRSAPredictor":
        """Loads complete predictor from saved checkpoint with strict=False compatibility."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        registry = DynamicElementRegistry()
        registry.symbol_to_id = checkpoint["registry_symbol_to_id"]
        registry.id_to_symbol = checkpoint["registry_id_to_symbol"]

        model = Causal_MCRSA_EGNN(
            max_elements=max(118, registry.total_elements + 10),
            hidden_dim=96,
            radial_dim=32,
            angular_dim=16,
            use_pbc=checkpoint.get("use_pbc", True),
            use_3body=checkpoint.get("use_3body", True)
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return cls(
            model=model,
            registry=registry,
            target_mean=checkpoint["target_mean"],
            target_std=checkpoint["target_std"],
            device=device
        )
