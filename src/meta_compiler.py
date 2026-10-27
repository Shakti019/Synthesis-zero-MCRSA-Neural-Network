from src.physics_constants import PRECURSORS_A, PRECURSORS_B

class MetaCompositionCompiler:
    def __init__(self):
        self.precursors_A = PRECURSORS_A
        self.precursors_B = PRECURSORS_B

    def compile_synthesis_path(self, eval_data):
        formula = eval_data['formula']
        A = eval_data['elem_A']
        B = eval_data['elem_B']
        
        prec_A = self.precursors_A.get(A, f"{A} Oxide")
        prec_B = self.precursors_B.get(B, f"{B} Oxide")
        
        temp = int(900 + 200 * eval_data['tolerance_t'] + 50 * max(0, eval_data['estimated_Ef_eV']))
        reaction_str = f"{prec_A} + {prec_B} --({temp}°C)--> {formula} + CO2↑"
        
        return {
            'target': formula,
            'precursor_A': prec_A,
            'precursor_B': prec_B,
            'calcination_temp_C': temp,
            'reaction_equation': reaction_str
        }
