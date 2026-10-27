# SYNTHESIS-ZERO
### Autonomous Material Intelligence & 3D Atomic Synthesis Architecture
> **Meta-Conscious Relational-Synthesis Architecture (MCRSA)**  
> A first-principles, geometry-aware deep learning framework that discovers stable crystal materials, predicts thermodynamic properties, compiles real-world synthesis pathways, and reconstructs 3D atomic lattices from real-world products.

---

## 🔬 1. Project Overview & Vision

Traditional Large Language Models (LLMs) hallucinate chemical formulas and invalid molecular bonds because they rely solely on text token prediction. Conversely, Density Functional Theory (DFT) quantum simulations are computationally expensive and cannot autonomously explore vast chemical spaces.

**SYNTHESIS-ZERO** bridges this gap:
1. **Geometric Graph Deep Learning**: Encodes quantum and structural atomic physics into an **Equivariant Graph Neural Network (`MCRSA_EGNN`)** that preserves 3D spatial rotation and translation invariances.
2. **First-Principles Thermodynamic Guardrails**: Validates candidates using physical constants (Shannon ionic radii, Pauling electronegativities, Goldschmidt tolerance factors $t$, and octahedral ratios $\mu$).
3. **Autonomous Active Exploration**: An **Entropy Curiosity Hunger Engine** continuously scans unexplored composition spaces, prioritizes high-uncertainty candidates, and conducts continuous online model updates.
4. **Synthesis Pathway Compiler**: Automatically generates executable solid-state calcination reaction recipes and precursor requirements.
5. **Real-World Camera-to-Atom Vision (In Development)**: An integrated mobile companion scanner where a user points their phone camera at any physical product or material (carbon, metal, plastic, ceramic), transferring the photo to the backend to reconstruct and render an interactive 3D atomic lattice on both phone and desktop.

---

## 📊 2. Current Project Stage: Where We Are

The project is currently at **Stage 3: A Fully Operational Research Prototype with an Active Deep Learning Engine**.

| Metric | Current Value | Status |
| :--- | :--- | :--- |
| **Project Stage** | **Stage 3 (Operational Prototype)** | Transitioning to Stage 4 (Productized App) |
| **Core Model** | `MCRSA_EGNN` (PyTorch) | Pre-trained & actively training online |
| **Pretraining Status** | **Completed (8 / 8 Epochs)** | Verified on 18,928 3D perovskite crystals |
| **Total Gradient Steps** | **7,456+ steps** | Continuous active online loop |
| **Discovered Vault** | **2,190+ materials** | Stored in memory vault and CSV archives |
| **Current Model Loss** | **~0.0324 (MSE)** | Stable convergence with validation gating |
| **Web Dashboard** | **Live Flask Application** | Real-time telemetry, Plotly visualizer, REST API |
| **External Grounding** | **Materials Project API** | Live background synchronization & ingestion pipeline |

---

## 🧠 3. Deep Learning Architecture (`MCRSA_EGNN`)

Located in [`src/models/mcrsa_egnn.py`](src/models/mcrsa_egnn.py), the neural network is a 4-layer residual geometry-aware graph network designed for atomic crystal structures.

```
                  [Atomic Composition + 3D Coordinates]
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
    [Node Features (8D)]                     [Global Features (6D)]
   • Shannon Ionic Radius                   • Mean Pairwise Distance
   • Pauling Electronegativity              • Std Pairwise Distance
   • Valence / Oxidation State              • Unit Cell Scale
   • Atomic Mass                            • Compositional Entropy
   • Atomic Number                          • Cation/Anion Radius Ratio
   • Periodic Group & Period                • Mean Atomic Number
   • Radius-to-Mass Ratio                               │
              │                                         │
              ▼                                         │
   [Linear Projection ➔ 96D]                            │
              │                                         │
              ▼                                         │
   ┌──────────────────────────────────────────┐         │
   │ 4x Residual Message Blocks               │         │
   │  • Interatomic Euclidean distance: ||Δx||²│        │
   │  • Edge MLP gating with SiLU & LayerNorm │         │
   │  • Node state updates & residual skip    │         │
   └──────────────────────────────────────────┘         │
              │                                         │
              ▼                                         │
    [Self-Attention Pooling]                            │
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                   [Graph Projection (96D + 6D)]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       [Formation Energy Head]         [Uncertainty Head]
      Predicted ΔEf (eV/atom)         Aleatoric Variance (σ²)
```

### Key Neural Network Features:
* **Distance-Aware Message Passing**: Edges between atoms compute spatial Euclidean displacement vectors $\Delta x_{ij} = x_i - x_j$ and squared distances $||\Delta x||^2$, ensuring that atomic bonds and geometry directly influence message propagation.
* **Dual-Head Readout**:
  * **Formation Energy ($\Delta E_f$)**: Predicts thermodynamic stability in electron-volts per atom.
  * **Aleatoric Uncertainty ($\sigma^2$)**: Predicts confidence using a softplus activation head, driving active-learning acquisition functions to explore high-uncertainty chemical domains.
* **Latent Representation ($z \in \mathbb{R}^{48}$)**: 48-dimensional continuous material embedding used for clustering, similarity search, and inverse material design.

---

## 🧪 4. Physical Constraints & Autonomous Engine

The machine learning model works in tandem with physical validation and autonomous search engines:

1. **Goldschmidt Tolerance & Octahedral Factors** ([`src/thermodynamic_validator.py`](src/thermodynamic_validator.py)):
   $$t = \frac{r_A + r_X}{\sqrt{2}(r_B + r_X)}, \quad \mu = \frac{r_B}{r_X}$$
   Filters out geometrically impossible crystal lattices before committing model predictions.
2. **Entropy Curiosity Hunger Engine** ([`src/entropy_hunger.py`](src/entropy_hunger.py)):
   Computes Shannon compositional entropy over ionic radii to quantify uncertainty and steer the autonomous generator toward unexplored element combinations.
3. **Synthesis Pathway Compiler** ([`src/meta_compiler.py`](src/meta_compiler.py)):
   Computes reaction energetics, calculates optimal calcination temperatures based on tolerance factors and formation energies, and produces real-world precursor chemical equations (e.g. $\text{BaCO}_3 + \text{TiO}_2 \xrightarrow{1120^\circ\text{C}} \text{BaTiO}_3 + \text{CO}_2\uparrow$).
4. **Materials Project Audited Ingestion** ([`src/data_ingestion/`](src/data_ingestion/)):
   Fetches verified Density Functional Theory (DFT) entries from the Materials Project API, deduplicates them via cryptographic fingerprints, and admits valid structures into the training corpus.

---

## 📱 5. The Real-World Vision: Phone Camera to 3D Atoms

A core objective of the project is building an **instant camera application** where any physical product or material can be scanned with a smartphone camera to reconstruct its 3D atomic structure:

```
[Phone Camera] ──(Snap Photo)──► [Vision AI Identification]
                                         │
                                         ▼ (Detects: Material & Formula)
                           [Synthesis Zero GNN Engine]
                                         │
                                         ▼ (Generates: 3D Unit Cell & Coordinates)
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   [Phone Screen 3D View]                                  [Desktop Dashboard 3D View]
   Interactive 3D ball-and-stick                           High-resolution lattice view,
   with touch orbit & atom inspector                       telemetry & synthesis pathway
```

### Coverage Across Real-World Materials:
* **Carbon Allotropes**: Diamond ($sp^3$ tetrahedral), Graphite ($sp^2$ planar sheets), Graphene, Carbon Nanotubes.
* **Metals & Alloys**: Aluminum, Copper, Gold, Iron/Steel (Face-Centered Cubic, Body-Centered Cubic lattices).
* **Plastics & Polymers**: PET water bottles, Polyethylene, Polystyrene chains.
* **Ceramics & Minerals**: Glass/Quartz ($\text{SiO}_2$), Table Salt ($\text{NaCl}$), Alumina ($\text{Al}_2\text{O}_3$), Perovskites ($\text{SrTiO}_3, \text{BaTiO}_3$).

---

## 📂 6. Repository Structure

```
Synthesis zero/
├── Data/                                    # Datasets, checkpoints, and archives
│   ├── perovskites_3d_dataset.json          # Primary pretraining dataset (18.9k structures)
│   ├── mcrsa_checkpoint.pt                  # Active PyTorch model checkpoint
│   ├── mcrsa_state.json                     # Training step counters, loss, and cycles
│   ├── mcrsa_memory_vault.json              # 2,190+ discovered material candidates
│   └── 05_mcrsa_feature_transformation.csv  # Flattened feature audit export
├── Experiment Notebook/                     # Interactive research notebooks
│   ├── Synthesis_Zero_MCRSA.ipynb           # Architecture PoC & mathematical foundations
│   └── model_api_experiment.ipynb          # Materials Project inference experiments
├── src/                                     # Modular core Python library
│   ├── acquisition/                         # Bayesian & uncertainty acquisition functions
│   ├── candidate_generation/                # Dynamic composition & valence generator
│   ├── data_ingestion/                      # Materials Project live sync & audit pipeline
│   ├── features/                            # 8D node & 6D global feature extraction
│   ├── memory/                              # External feedback & memory vault manager
│   ├── models/                              # PyTorch MCRSA_EGNN network implementation
│   ├── validation/                          # Physical proposal classifiers
│   ├── consciousness_runtime.py             # Dynamic execution registry
│   ├── entropy_hunger.py                    # Shannon entropy curiosity engine
│   ├── meta_compiler.py                     # Solid-state calcination compiler
│   ├── physics_constants.py                 # Shannon radii, electronegativity & precursors
│   └── thermodynamic_validator.py           # Goldschmidt tolerance & stability filters
├── static/                                  # Web frontend dashboard
│   ├── app.js                               # Dashboard controller & telemetry polling
│   ├── index.html                           # Single-page research console UI
│   └── style.css                            # Modern scientific dark-mode stylesheet
├── .env                                     # Environment secrets (MP_API_KEY)
├── app.py                                   # Flask web server & REST API
├── discovery_engine.py                      # Autonomous discovery worker & loop manager
└── README.md                                # Project documentation
```

---

## 🚀 7. Quickstart Guide

### Prerequisites
* Python 3.10+ (tested with Python 3.12/3.13)
* PyTorch (`torch`)
* Flask, NumPy, Pandas, Plotly

### Installation
1. Clone or navigate to the project directory:
   ```powershell
   cd "c:\Users\shakti singh\OneDrive\Desktop\Synthesis zero"
   ```
2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Set your Materials Project API key in `.env`:
   ```env
   MP_API_KEY=your_api_key_here
   ```

### Running the Application
Launch the autonomous engine and web dashboard:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

### Key API Endpoints
* `GET /api/status`: Engine running status, total steps, current formula, and loss.
* `GET /api/discoveries?limit=100`: List of discovered materials with calculated tolerance and synthesis pathways.
* `GET /api/telemetry`: Model evaluation pulse (MAE, RMSE, $R^2$, parameter weights, log stream).
* `GET /api/neural_graph`: Active neural node and edge activations for 3D visualization.
* `GET /api/lattice_3d/<formula>`: 3D atomic coordinates and elemental composition for any formula.
* `POST /api/control/start`: Start/resume the autonomous discovery loop.
* `POST /api/control/pause`: Pause the engine and persist checkpoints to disk.

---

## 🗺️ 8. Remaining Roadmap to Full Completion

```
[Milestone 1: Camera & 3D Viewer]  ──►  [Milestone 2: Universal Materials]
 • Mobile QR Code Connect                • Carbon allotropes (Diamond/Graphite)
 • Camera snapshot upload endpoint       • Metals (Al, Cu, Fe, Au)
 • Three.js 60fps 3D atomic viewer       • Plastics & Polymers (PET, PE)
 
                  │
                  ▼
[Milestone 3: Vision AI Classifier] ──► [Milestone 4: PWA & Multi-Property GNN]
 • Automatic material recognition        • Bandgap, density & conductivity heads
 • Chemical formula resolution           • Mobile PWA (Add to Home Screen)
 • Confidence scores & alternatives      • CIF/XYZ/GLTF 3D structure export
```

---

## 📜 License & Acknowledgments
* **Framework**: Built with PyTorch, Flask, and Plotly.
* **Data Sources**: Pretrained on Perovskite crystallographic datasets and grounded against the **Materials Project** API.
