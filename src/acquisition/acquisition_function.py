"""Uncertainty-aware active-learning candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AcquisitionScore:
    energy: float
    uncertainty: float
    novelty: float
    score: float


class AcquisitionFunction:
    """Balance low predicted energy, uncertainty, and chemical novelty."""

    def score(
        self,
        predicted_energy: float,
        uncertainty: float,
        formula: str,
        seen_formulas: set[str],
    ) -> AcquisitionScore:
        novelty = 1.0 if formula not in seen_formulas else 0.0
        safe_energy = predicted_energy if isfinite(predicted_energy) else 1e6
        safe_uncertainty = max(0.0, uncertainty) if isfinite(uncertainty) else 0.0
        score = -safe_energy + 0.35 * safe_uncertainty + 0.15 * novelty
        return AcquisitionScore(safe_energy, safe_uncertainty, novelty, score)
