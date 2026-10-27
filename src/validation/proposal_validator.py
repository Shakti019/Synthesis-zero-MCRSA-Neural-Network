"""Validation gates for predicted proposals.

This module deliberately does not claim experimental or DFT validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ProposalValidation:
    status: str
    physically_stable: bool
    requires_external_validation: bool


class ProposalValidator:
    def classify(self, tolerance: float, predicted_energy: float) -> ProposalValidation:
        geometry_valid = 0.81 <= tolerance <= 1.05
        finite_prediction = isfinite(predicted_energy)
        physically_stable = geometry_valid and finite_prediction and predicted_energy < 0.6
        if physically_stable:
            return ProposalValidation("physically_stable_requires_validation", True, True)
        if geometry_valid and finite_prediction:
            return ProposalValidation("model_ranked_requires_validation", False, True)
        return ProposalValidation("rejected", False, False)
