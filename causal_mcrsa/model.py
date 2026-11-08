"""
Causal MCRSA Neural Network Architecture.
Implements 3-Body Equivariant Relational Message Passing with Decoupled Physical Branches:
Net Energy ΔEf = E_chemical + E_strain, where E_strain >= 0 is strictly enforced via Softplus.
Supports Periodic Boundary Conditions (PBC) and modular ablations.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import CausalReasonExtractor


class EquivariantRelationalBlock(nn.Module):
    """
    Relational message passing block with combined 2-body radial
    and 3-body angular geometric gating.
    """
    def __init__(self, hidden_dim: int, geom_dim: int = 48):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + geom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h: torch.Tensor, geom_rbf: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        hi = h.unsqueeze(1).repeat(1, N, 1)
        hj = h.unsqueeze(0).repeat(N, 1, 1)

        edge_input = torch.cat([hi, hj, geom_rbf], dim=-1)
        gates = self.edge_mlp(edge_input)
        messages = torch.sum(gates * hj, dim=1) / max(N, 1)

        return h + self.node_update(torch.cat([h, messages], dim=-1))


class Causal_MCRSA_EGNN(nn.Module):
    """
    High-level causal neural network with 3-body geometric reasoning,
    periodic boundary conditions, and decoupled physical enthalpy and strain branches.
    Zero hardcoded dictionaries. Fully learnable dynamic element embeddings.
    """
    def __init__(
        self,
        max_elements: int = 118,
        hidden_dim: int = 96,
        radial_dim: int = 32,
        angular_dim: int = 16,
        latent_dim: int = 48,
        use_pbc: bool = True,
        use_3body: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_pbc = use_pbc
        self.use_3body = use_3body
        self.angular_dim = angular_dim if use_3body else 0
        geom_dim = radial_dim + self.angular_dim

        self.reason_extractor = CausalReasonExtractor(
            num_radial=radial_dim,
            num_angular=angular_dim if use_3body else 4,
            use_pbc=use_pbc
        )

        # Learnable element embedding space (initialized with physical variance)
        self.element_embeddings = nn.Embedding(max_elements, hidden_dim)
        nn.init.normal_(self.element_embeddings.weight, mean=0.0, std=0.15)

        # 3x Equivariant Relational Blocks
        self.block1 = EquivariantRelationalBlock(hidden_dim, geom_dim)
        self.block2 = EquivariantRelationalBlock(hidden_dim, geom_dim)
        self.block3 = EquivariantRelationalBlock(hidden_dim, geom_dim)

        # Self-Attention Graph Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Causal Fusion layer (fuses crystal graph with coordinate strain reason)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        # --- DECOUPLED CAUSAL REASONING BRANCHES ---
        # Branch 1: Chemical Bonding Affinity (electron exchange enthalpy)
        self.branch_chemical = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 1)
        )
        # Branch 2: Geometric Steric Strain Penalty (STRICTLY NON-NEGATIVE via Softplus)
        self.branch_strain_core = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
            nn.Softplus()  # Physics constraint: geometric strain penalty is always >= 0
        )
        # Branch 3: Aleatoric Uncertainty Head (confidence estimation)
        self.branch_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(
        self,
        element_ids: torch.Tensor,
        coords: torch.Tensor,
        lattice_a: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Compute spatial 2-body radial + 3-body angular features and coordinate strain
        geom_rbf, dists, strain = self.reason_extractor(coords, lattice_a=lattice_a)

        # If 3-body disabled for ablation, use only radial features
        if not self.use_3body:
            geom_rbf = geom_rbf[:, :, :32]

        # 2. Dynamic element embedding lookup
        h = self.element_embeddings(element_ids)  # [N, hidden_dim]

        # 3. Equivariant Message Passing
        h = self.block1(h, geom_rbf)
        h = self.block2(h, geom_rbf)
        h = self.block3(h, geom_rbf)

        # 4. Self-Attention Pooling
        attn_weights = F.softmax(self.attention_pool(h), dim=0)  # [N, 1]
        pooled = torch.sum(attn_weights * h, dim=0)  # [hidden_dim]

        # 5. Fuse crystal state with explicit coordinate strain reason
        fused = self.fusion(torch.cat([pooled, strain], dim=-1))  # [hidden_dim]
        latent_z = self.latent_proj(fused)  # [latent_dim]

        # 6. Decoupled Causal Predictions
        e_chemical = self.branch_chemical(fused)
        e_strain = self.branch_strain_core(fused)  # Always >= 0
        net_formation_energy = e_chemical + e_strain  # Total Normalized Energy
        uncertainty = self.branch_uncertainty(fused)

        return {
            "net_energy_Ef": net_formation_energy,
            "chemical_energy": e_chemical,
            "steric_strain": e_strain,
            "uncertainty": uncertainty,
            "latent_z": latent_z,
            "coordinate_strain_reason": strain,
            "dists": dists
        }
