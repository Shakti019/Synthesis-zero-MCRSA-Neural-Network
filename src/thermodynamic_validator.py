import math
from src.physics_constants import ELEMENT_PHYSICS

class PhysicalThermodynamicValidator:
    def __init__(self):
        self.physics = ELEMENT_PHYSICS

    def compute_goldschmidt_tolerance(self, elem_A, elem_B, elem_X):
        rA = self.physics[elem_A]['radius']
        rB = self.physics[elem_B]['radius']
        rX = self.physics[elem_X]['radius']
        
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        return t, mu

    def evaluate_candidate(self, elem_A, elem_B, elem_X, estimated_Ef):
        t, mu = self.compute_goldschmidt_tolerance(elem_A, elem_B, elem_X)
        is_stable = (0.81 <= t <= 1.05) and (estimated_Ef < 0.6)
        
        return {
            'formula': f"{elem_A}{elem_B}{elem_X}3",
            'elem_A': elem_A,
            'elem_B': elem_B,
            'elem_X': elem_X,
            'tolerance_t': round(t, 4),
            'octahedral_mu': round(mu, 4),
            'estimated_Ef_eV': round(estimated_Ef, 4),
            'is_physically_stable': is_stable
        }
