"""
Causal Reason Extractor for Causal MCRSA Architecture.
Extracts continuous rotation-invariant 2-body radial distance kernels (r_ij)
and 3-body directional bond angle kernels (theta_jik) directly from 3D Cartesian coordinates.
Features full Periodic Boundary Conditions (PBC) handling with 27-cell periodic neighbor shells
and minimum image convention for crystal lattices.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class CausalReasonExtractor(nn.Module):
    """
    Extracts continuous rotation-invariant 2-body radial and 3-body angular
    geometric features directly from 3D Cartesian coordinates with Periodic Boundary Conditions (PBC).
    Captures complete crystal coordination shells (e.g., 6-fold BX6 octahedron, 12-fold AX12 cuboctahedron).
    """
    def __init__(
        self,
        num_radial: int = 32,
        num_angular: int = 16,
        r_min: float = 0.5,
        r_max: float = 5.0,
        use_pbc: bool = True
    ):
        super().__init__()
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.use_pbc = use_pbc
        self.r_max = r_max
        self.r_min = r_min

        # Continuous Gaussian RBF centers for radial pair distances (0.5 to 5.0 Angstroms)
        radial_centers = torch.linspace(r_min, r_max, num_radial)
        self.register_buffer("radial_centers", radial_centers)
        self.radial_gamma = 1.0 / (((r_max - r_min) / num_radial) ** 2)

        # Continuous Gaussian RBF centers for 3-body bond angles (-1.0 to 1.0 in cos theta)
        angle_centers = torch.linspace(-1.0, 1.0, num_angular)
        self.register_buffer("angle_centers", angle_centers)
        self.angle_gamma = 1.0 / ((2.0 / num_angular) ** 2)

        # Base 27 periodic shift multipliers: dx, dy, dz in {-1, 0, 1}
        shifts = []
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                for dz in (-1.0, 0.0, 1.0):
                    shifts.append([dx, dy, dz])
        self.register_buffer("base_shifts", torch.tensor(shifts, dtype=torch.float32))

    def forward(
        self,
        coords: torch.Tensor,
        lattice_a: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: Cartesian coordinates tensor of shape [N, 3] in Angstroms.
            lattice_a: Optional cubic lattice constant in Angstroms. If None, inferred as 2 * coords[1, 1].
        Returns:
            geom_rbf: Combined 2-body radial + 3-body angular tensor of shape [N, N, num_radial + num_angular]
            dists: Effective pair distance tensor of shape [N, N, 1]
            strain: Coordinate strain scalar tensor of shape [1]
        """
        N = coords.shape[0]
        device = coords.device

        if self.use_pbc:
            # Infer cubic lattice constant a: in 5-atom perovskites, a = 2 * coords[1, 1]
            if lattice_a is None:
                a_val = torch.clamp(2.0 * coords[1, 1].abs(), min=2.5, max=7.0)
            else:
                a_val = lattice_a

            # 1. Periodic Shifts: 27 unit cell shifts [27, 3]
            periodic_shifts = self.base_shifts.to(device) * a_val

            # Pairwise displacements across 27 periodic image cells: [N, N, 27, 3]
            # disp_all[i, j, k] = (coords[j] + shifts[k]) - coords[i]
            disp_all = (coords.unsqueeze(0).unsqueeze(2) + periodic_shifts.unsqueeze(0).unsqueeze(0)) - coords.unsqueeze(1).unsqueeze(2)
            dists_all = torch.norm(disp_all, dim=-1, keepdim=True)  # [N, N, 27, 1]

            # Mask out self-interaction (d < 0.05) and interactions beyond cutoff r_max
            valid_mask = (dists_all > 0.05) & (dists_all <= self.r_max)  # [N, N, 27, 1]

            # 2-Body Radial RBF across all periodic coordination neighbors: [N, N, 27, 32]
            rbf_all = torch.exp(-self.radial_gamma * (dists_all - self.radial_centers) ** 2)
            radial_rbf = torch.sum(rbf_all * valid_mask.float(), dim=2)  # [N, N, 32]

            # Minimum image distance for primary edge distances: [N, N, 1]
            disp_min = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 3]
            disp_min_pbc = disp_min - a_val * torch.round(disp_min / a_val)
            dists = torch.norm(disp_min_pbc, dim=-1, keepdim=True)

            # 2. 3-Body Directional Bond Angles under Minimum Image Convention
            norm_disp = disp_min_pbc / (dists + 1e-8)
            cos_theta = torch.einsum("ijd,ikd->ijk", norm_disp, norm_disp)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angular_rbf = torch.exp(-self.angle_gamma * (cos_theta.unsqueeze(-1) - self.angle_centers) ** 2)

            # Mask out self-triplets
            mask_ijk = torch.ones(N, N, N, 1, device=device)
            for idx in range(N):
                mask_ijk[idx, idx, :, :] = 0.0
                mask_ijk[idx, :, idx, :] = 0.0
                mask_ijk[:, idx, idx, :] = 0.0

            angular_rbf = angular_rbf * mask_ijk
            edge_angles = torch.sum(angular_rbf, dim=2) / max(N - 2, 1)  # [N, N, 16]

            # 3. Steric Strain: Deviation from ideal cubic perovskite geometry
            # Ideal octahedral B-X distance is a/2
            bx_dist = torch.norm(disp_min_pbc[1, 2:], dim=-1)  # distances from B to X1, X2, X3
            strain = torch.mean(torch.abs(bx_dist - (a_val / 2.0))).unsqueeze(0)

        else:
            # Non-PBC fallback (isolated cluster mode for ablation comparison)
            disp = coords.unsqueeze(1) - coords.unsqueeze(0)
            dists = torch.norm(disp, dim=-1, keepdim=True)
            radial_rbf = torch.exp(-self.radial_gamma * (dists - self.radial_centers) ** 2)

            norm_disp = disp / (dists + 1e-8)
            cos_theta = torch.einsum("ijd,ikd->ijk", norm_disp, norm_disp)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angular_rbf = torch.exp(-self.angle_gamma * (cos_theta.unsqueeze(-1) - self.angle_centers) ** 2)

            mask_ijk = torch.ones(N, N, N, 1, device=device)
            for idx in range(N):
                mask_ijk[idx, idx, :, :] = 0.0
                mask_ijk[idx, :, idx, :] = 0.0
                mask_ijk[:, idx, idx, :] = 0.0

            angular_rbf = angular_rbf * mask_ijk
            edge_angles = torch.sum(angular_rbf, dim=2) / max(N - 2, 1)

            centroid = torch.mean(coords, dim=0, keepdim=True)
            strain = torch.mean(torch.norm(coords - centroid, dim=-1, keepdim=True), dim=0)

        geom_rbf = torch.cat([radial_rbf, edge_angles], dim=-1)  # [N, N, 48]
        return geom_rbf, dists, strain
