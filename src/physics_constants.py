import os

DATA_DIR = "Data"
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join("..", "Data")
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(DATA_DIR, "mcrsa_checkpoint.pt")
MEMORY_VAULT_PATH = os.path.join(DATA_DIR, "mcrsa_memory_vault.json")
STATE_PATH = os.path.join(DATA_DIR, "mcrsa_state.json")
LOG_HISTORY_PATH = os.path.join(DATA_DIR, "mcrsa_live_logs.json")

# Core Physics Properties: Shannon/Covalent Radii (Å), Pauling Electronegativity (χ), Valence State, Mass (amu)
ELEMENT_PHYSICS = {
    # Organics, Polymers & Life Elements
    'C':  {'radius': 0.77, 'electronegativity': 2.55, 'valence': 4, 'mass': 12.011},
    'H':  {'radius': 0.37, 'electronegativity': 2.20, 'valence': 1, 'mass': 1.008},
    'O':  {'radius': 1.40, 'electronegativity': 3.44, 'valence': -2, 'mass': 16.00},
    'N':  {'radius': 1.46, 'electronegativity': 3.04, 'valence': -3, 'mass': 14.01},
    'P':  {'radius': 1.10, 'electronegativity': 2.19, 'valence': 5, 'mass': 30.974},
    'S':  {'radius': 1.84, 'electronegativity': 2.58, 'valence': -2, 'mass': 32.06},
    'Cl': {'radius': 0.99, 'electronegativity': 3.16, 'valence': -1, 'mass': 35.45},
    'F':  {'radius': 1.33, 'electronegativity': 3.98, 'valence': -1, 'mass': 19.00},
    'Br': {'radius': 1.14, 'electronegativity': 2.96, 'valence': -1, 'mass': 79.904},
    'I':  {'radius': 1.33, 'electronegativity': 2.66, 'valence': -1, 'mass': 126.90},

    # Structural & Commercial Metals
    'Al': {'radius': 0.535, 'electronegativity': 1.61, 'valence': 3, 'mass': 26.98},
    'Fe': {'radius': 0.645, 'electronegativity': 1.83, 'valence': 3, 'mass': 55.85},
    'Cu': {'radius': 0.73,  'electronegativity': 1.90, 'valence': 2, 'mass': 63.55},
    'Ni': {'radius': 0.69,  'electronegativity': 1.91, 'valence': 2, 'mass': 58.69},
    'Ti': {'radius': 0.605, 'electronegativity': 1.54, 'valence': 4, 'mass': 47.87},
    'Zn': {'radius': 0.74,  'electronegativity': 1.65, 'valence': 2, 'mass': 65.38},
    'Mg': {'radius': 0.72,  'electronegativity': 1.31, 'valence': 2, 'mass': 24.305},
    'Cr': {'radius': 0.615, 'electronegativity': 1.66, 'valence': 3, 'mass': 51.996},
    'Mn': {'radius': 0.645, 'electronegativity': 1.55, 'valence': 4, 'mass': 54.94},
    'Co': {'radius': 0.65,  'electronegativity': 1.88, 'valence': 2, 'mass': 58.933},
    'Mo': {'radius': 0.65,  'electronegativity': 2.16, 'valence': 4, 'mass': 95.95},
    'W':  {'radius': 0.62,  'electronegativity': 2.36, 'valence': 4, 'mass': 183.84},
    'V':  {'radius': 0.54,  'electronegativity': 1.63, 'valence': 5, 'mass': 50.942},
    'Au': {'radius': 0.85,  'electronegativity': 2.54, 'valence': 1, 'mass': 196.97},
    'Ag': {'radius': 1.15,  'electronegativity': 1.93, 'valence': 1, 'mass': 107.87},
    'Pt': {'radius': 0.80,  'electronegativity': 2.28, 'valence': 2, 'mass': 195.08},
    'Pb': {'radius': 1.19,  'electronegativity': 2.33, 'valence': 2, 'mass': 207.2},

    # Semiconductors, Minerals & Ceramics
    'Si': {'radius': 0.40,  'electronegativity': 1.90, 'valence': 4, 'mass': 28.085},
    'B':  {'radius': 0.27,  'electronegativity': 2.04, 'valence': 3, 'mass': 10.81},
    'Ge': {'radius': 0.53,  'electronegativity': 2.01, 'valence': 4, 'mass': 72.63},
    'Sn': {'radius': 0.69,  'electronegativity': 1.96, 'valence': 4, 'mass': 118.71},
    'Ga': {'radius': 0.62,  'electronegativity': 1.81, 'valence': 3, 'mass': 69.723},
    'In': {'radius': 0.80,  'electronegativity': 1.78, 'valence': 3, 'mass': 114.82},
    'Bi': {'radius': 1.03,  'electronegativity': 2.02, 'valence': 3, 'mass': 208.98},

    # Alkali & Alkaline Earth Elements
    'Li': {'radius': 0.76,  'electronegativity': 0.98, 'valence': 1, 'mass': 6.94},
    'Na': {'radius': 1.02,  'electronegativity': 0.93, 'valence': 1, 'mass': 22.99},
    'K':  {'radius': 1.38,  'electronegativity': 0.82, 'valence': 1, 'mass': 39.10},
    'Cs': {'radius': 1.67,  'electronegativity': 0.79, 'valence': 1, 'mass': 132.91},
    'Ca': {'radius': 1.00,  'electronegativity': 1.00, 'valence': 2, 'mass': 40.08},
    'Sr': {'radius': 1.18,  'electronegativity': 0.95, 'valence': 2, 'mass': 87.62},
    'Ba': {'radius': 1.35,  'electronegativity': 0.89, 'valence': 2, 'mass': 137.33},
    'La': {'radius': 1.03,  'electronegativity': 1.10, 'valence': 3, 'mass': 138.91},
    'Zr': {'radius': 0.72,  'electronegativity': 1.33, 'valence': 4, 'mass': 91.22},
    'Hf': {'radius': 0.71,  'electronegativity': 1.30, 'valence': 4, 'mass': 178.49},
    'Nb': {'radius': 0.64,  'electronegativity': 1.60, 'valence': 5, 'mass': 92.91},
    'Ta': {'radius': 0.64,  'electronegativity': 1.50, 'valence': 5, 'mass': 180.95},
}

PRECURSORS_A = {
    'Sr': 'SrCO3', 'Ba': 'BaCO3', 'Ca': 'CaCO3', 'La': 'La2O3',
    'K': 'K2CO3', 'Cs': 'Cs2CO3', 'Na': 'Na2CO3', 'Li': 'Li2CO3',
}
PRECURSORS_B = {
    'Ti': 'TiO2', 'Zr': 'ZrO2', 'Hf': 'HfO2', 'Fe': 'Fe2O3',
    'Mn': 'MnO2', 'Nb': 'Nb2O5', 'Ta': 'Ta2O5', 'Si': 'SiO2', 'Al': 'Al2O3',
}

