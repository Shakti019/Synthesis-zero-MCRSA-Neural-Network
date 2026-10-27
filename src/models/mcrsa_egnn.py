"""Residual geometry-aware graph model for material energy prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_states: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        count = node_states.shape[0]
        delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
        distance_sq = (delta * delta).sum(dim=-1, keepdim=True)
        source = node_states.unsqueeze(1).expand(count, count, -1)
        target = node_states.unsqueeze(0).expand(count, count, -1)
        gates = self.edge_mlp(torch.cat([source, target, distance_sq], dim=-1))
        messages = gates * target
        aggregated = messages.sum(dim=1) / max(count, 1)
        return self.norm(node_states + self.update(torch.cat([node_states, aggregated], dim=-1)))


class MCRSA_EGNN(nn.Module):
    """Multi-layer graph predictor with uncertainty and latent representations."""

    def __init__(
        self,
        in_features: int = 8,
        global_features: int = 6,
        hidden_dim: int = 96,
        latent_dim: int = 48,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.global_features = global_features
        self.node_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(ResidualMessageBlock(hidden_dim) for _ in range(layers))
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim + global_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
        )
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        coords_3d: torch.Tensor,
        global_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = self.node_proj(node_feats)
        for block in self.blocks:
            states = block(states, coords_3d)
        weights = torch.softmax(self.attention_pool(states).squeeze(-1), dim=0).unsqueeze(-1)
        pooled = (states * weights).sum(dim=0)
        if global_feats is None:
            global_feats = torch.zeros(self.global_features, device=pooled.device)
        graph = self.graph_proj(torch.cat([pooled, global_feats], dim=-1))
        latent = self.latent_head(graph)
        energy_and_log_variance = self.energy_head(graph)
        energy = energy_and_log_variance[:1]
        uncertainty = F.softplus(energy_and_log_variance[1:])
        return energy, latent, uncertainty
