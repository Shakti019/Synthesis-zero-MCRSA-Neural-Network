class ConsciousnessBodyRuntime:
    def __init__(self):
        self.registry = {}

    def register_compiled_discovery(self, target_formula, synthesis_meta):
        fn_name = f"synthesize_{target_formula}"
        code_str = f'''
def {fn_name}():
    precursors = ["{synthesis_meta['precursor_A']}", "{synthesis_meta['precursor_B']}"]
    temperature = {synthesis_meta['calcination_temp_C']}
    return "{target_formula} Solid State Powder synthesized at {synthesis_meta['calcination_temp_C']}°C"
'''
        exec(code_str, globals())
        self.registry[target_formula] = globals()[fn_name]
        return fn_name

    def execute(self, target_formula):
        if target_formula in self.registry:
            return self.registry[target_formula]()
        elif f"synthesize_{target_formula}" in globals():
            return globals()[f"synthesize_{target_formula}"]()
        else:
            return f"Function synthesize_{target_formula} not found in active consciousness registry."
