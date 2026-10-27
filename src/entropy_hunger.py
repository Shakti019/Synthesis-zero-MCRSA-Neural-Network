import numpy as np
from src.physics_constants import ELEMENT_PHYSICS

class EntropyCuriosityHungerEngine:
    def __init__(self, known_compositions=None):
        self.physics = ELEMENT_PHYSICS
        self.known_compositions = known_compositions if known_compositions is not None else set()

    def update_known_compositions(self, compositions_set):
        self.known_compositions = compositions_set

    def compute_composition_entropy(self, elements):
        radii = [self.physics[e]['radius'] for e in elements if e in self.physics]
        if not radii:
            return 0.0
        p = np.array(radii) / np.sum(radii)
        entropy = -np.sum(p * np.log2(p + 1e-9))
        return float(entropy)

    def evaluate_target_uncertainty(self, elements):
        sorted_comp = tuple(sorted(elements))
        is_known = sorted_comp in self.known_compositions
        entropy = self.compute_composition_entropy(elements)
        uncertainty = 1.8 if not is_known else 0.15
        h_score = entropy * uncertainty
        return h_score, not is_known
