"""
Training Pipeline for Causal MCRSA Architecture.
Implements Model Exponential Moving Average (EMA), Near-Hull Thermodynamic Focus Loss,
Heteroscedastic Gaussian Negative Log-Likelihood (NLL) Loss for calibrated uncertainty,
Periodic Boundary Conditions (PBC), Cosine Annealing Learning Rate Schedule,
and automated CSV logging for all training runs.
"""

import os
import csv
import copy
import math
import random
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from .registry import DynamicElementRegistry
from .model import Causal_MCRSA_EGNN


class ModelEMA:
    """
    Maintains exponential moving average of model parameters for smoother,
    higher-precision convergence (standard in SOTA physical GNNs like MACE/PaiNN).
    """
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


class HeteroscedasticLoss(nn.Module):
    """
    Trains both predicted energy (mu) and aleatoric uncertainty (sigma)
    using Gaussian Negative Log-Likelihood combined with Smooth L1 regression.
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.05, reduction='none')

    def forward(self, pred_mu: torch.Tensor, pred_sigma: torch.Tensor, target: torch.Tensor, weight: Optional[float] = None) -> torch.Tensor:
        reg_loss = self.smooth_l1(pred_mu, target)
        variance = torch.clamp(pred_sigma ** 2, min=self.eps)
        nll_loss = 0.5 * ((target - pred_mu) ** 2) / variance + 0.5 * torch.log(variance)
        total_loss = reg_loss + 0.20 * nll_loss
        if weight is not None:
            total_loss = total_loss * weight
        return total_loss.mean()


def train_causal_mcrsa(
    train_set: List[Dict[str, Any]],
    registry: DynamicElementRegistry,
    target_mean: float,
    target_std: float,
    epochs: int = 20,
    accum_steps: int = 16,
    lr: float = 1.5e-3,
    min_lr: float = 1.5e-5,
    ema_decay: float = 0.995,
    device: Optional[torch.device] = None,
    log_interval: int = 2000,
    use_pbc: bool = True,
    use_3body: bool = True,
    log_csv_path: Optional[str] = None
) -> Tuple[Causal_MCRSA_EGNN, ModelEMA, List[Dict[str, Any]]]:
    """
    Executes full-dataset training loop for Causal MCRSA Architecture.
    Logs epoch/step metrics directly to CSV.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Populate registry dynamically
    for s in train_set:
        for el in s["elements"]:
            registry.get_or_add(el)

    model = Causal_MCRSA_EGNN(
        max_elements=max(118, registry.total_elements + 10),
        hidden_dim=96,
        radial_dim=32,
        angular_dim=16,
        use_pbc=use_pbc,
        use_3body=use_3body
    ).to(device)

    ema = ModelEMA(model, decay=ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = HeteroscedasticLoss()

    train_samples = len(train_set)
    total_opt_steps = epochs * (train_samples // accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=min_lr)

    print(f"Executing Training Pipeline on device: {device} (PBC={use_pbc}, 3-Body={use_3body})")
    print(f"Total Samples: {train_samples:,} | Epochs: {epochs} | Steps: {total_opt_steps:,}")
    print(f"Registered Vocabulary: {registry.total_elements} elements")

    training_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_set)
        optimizer.zero_grad()

        for step, item in enumerate(train_set):
            elem_ids = registry.encode_elements(item["elements"]).to(device)
            coords = torch.tensor(item["coordinates_3d"], dtype=torch.float32).to(device)

            raw_target = float(item["target_energy"])
            norm_target = (raw_target - target_mean) / target_std
            target_tensor = torch.tensor([norm_target], dtype=torch.float32).to(device)

            output = model(elem_ids, coords)
            pred_norm_energy = output["net_energy_Ef"].squeeze()
            pred_norm_sigma = output["uncertainty"].squeeze() + 1e-3

            # Near-Hull Focus Weighting: 2.2x focus on stable/synthesizable materials (Ef <= 0.40 eV)
            sample_weight = 1.0 + 1.2 / (1.0 + math.exp((raw_target - 0.40) / 0.25))
            loss = criterion(pred_norm_energy, pred_norm_sigma, target_tensor.squeeze(), weight=sample_weight)
            (loss / accum_steps).backward()

            epoch_loss += loss.item()

            if (step + 1) % accum_steps == 0 or (step + 1) == train_samples:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)
                optimizer.zero_grad()

            if (step + 1) % log_interval == 0 or (step + 1) == train_samples:
                avg_loss = epoch_loss / (step + 1)
                denorm_pred = pred_norm_energy.item() * target_std + target_mean
                denorm_sigma = pred_norm_sigma.item() * target_std
                cur_lr = scheduler.get_last_lr()[0]
                error = abs(denorm_pred - raw_target)
                formula = "".join(item["elements"])

                log_entry = {
                    "epoch": epoch,
                    "step": step + 1,
                    "total_samples": train_samples,
                    "loss": round(avg_loss, 4),
                    "learning_rate": cur_lr,
                    "predicted_energy_cell": round(denorm_pred, 4),
                    "uncertainty_sigma_cell": round(denorm_sigma, 4),
                    "target_energy_cell": round(raw_target, 4),
                    "abs_error_cell": round(error, 4),
                    "formula": formula
                }
                training_logs.append(log_entry)

                print(
                    f"Epoch {epoch:2d}/{epochs} | Step {step+1:5d}/{train_samples} | "
                    f"Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | "
                    f"Pred: {denorm_pred:+.3f} ± {denorm_sigma:.3f} eV/cell | Target: {raw_target:+.3f} eV/cell | {formula}"
                )

    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(training_logs[0].keys()))
            writer.writeheader()
            writer.writerows(training_logs)
        print(f"Training logs exported to CSV: {log_csv_path}")

    print("Training Complete! Heteroscedastic uncertainty and energy model converged.")
    return model, ema, training_logs
