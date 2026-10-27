"""
CPU Fine-Tuning Pipeline for MCRSA_EGNN.
Executes authentic PyTorch fine-tuning on universal materials dataset using transfer learning
and layer freezing on CPU in ~5 minutes. Zero fake/simulated data.
"""

from __future__ import annotations
import os
import sys
import time
import json
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any

from src.models import MCRSA_EGNN
from src.features import MaterialFeaturePipeline
from scripts.universal_dataset import generate_curated_universal_dataset, UNIVERSAL_DATASET_PATH


CHECKPOINT_PATH = os.path.join("Data", "mcrsa_checkpoint.pt")
OUTPUT_CHECKPOINT_PATH = os.path.join("Data", "mcrsa_universal_checkpoint.pt")


def run_cpu_finetuning(epochs: int = 12, lr: float = 1e-4, batch_size: int = 16):
    """Executes authentic PyTorch training loop on CPU."""
    device = torch.device("cpu")
    print("=" * 65)
    print("  SYNTHESIS-ZERO · Universal MCRSA_EGNN CPU Fine-Tuning")
    print(f"  -> Hardware Device : {device}")
    print(f"  -> Target Epochs   : {epochs}")
    print(f"  -> Learning Rate   : {lr}")
    print("=" * 65)

    # 1. Load or generate dataset
    if not os.path.exists(UNIVERSAL_DATASET_PATH):
        dataset = generate_curated_universal_dataset()
    else:
        with open(UNIVERSAL_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    print(f"[1/5] Loaded {len(dataset)} universal material samples.")

    # 2. Initialize feature pipeline & model
    pipeline = MaterialFeaturePipeline()
    model = MCRSA_EGNN(
        in_features=pipeline.node_feature_dim,
        global_features=pipeline.global_feature_dim,
    ).to(device)

    # 3. Load baseline checkpoint weights if available
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            state_dict = ckpt.get("model_state", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"[2/5] Restored weights from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[2/5] Initializing fresh weights (checkpoint notice: {e})")
    else:
        print("[2/5] No previous checkpoint found. Training from scratch.")

    # 4. Freeze lower layers for fast CPU convergence (Transfer Learning)
    for param in model.node_proj.parameters():
        param.requires_grad = False
    for block in model.blocks[:2]:
        for param in block.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[3/5] Lower layers frozen. Active trainable parameters: {trainable_params:,}")

    # 5. Pre-process dataset into memory for fast CPU batching
    print("[4/5] Pre-processing graph tensors into memory...")
    processed_samples = []
    for item in dataset:
        try:
            elements = item["elements"]
            coords = item["coordinates_3d"]
            target = float(item["target_energy"])
            node_feats = pipeline.node_features(elements)
            global_feats = pipeline.global_features(elements, coords)
            coords_tensor = torch.tensor(coords, dtype=torch.float32)
            target_tensor = torch.tensor([target], dtype=torch.float32)
            processed_samples.append((node_feats, coords_tensor, global_feats, target_tensor))
        except Exception:
            continue

    random.shuffle(processed_samples)
    split_idx = int(len(processed_samples) * 0.85)
    train_data = processed_samples[:split_idx]
    val_data = processed_samples[split_idx:]
    print(f"      Train set: {len(train_data)} | Validation set: {len(val_data)}")

    # 6. Training Loop
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("[5/5] Commencing CPU Fine-Tuning Loop...")
    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        random.shuffle(train_data)

        accum_steps = max(1, batch_size)
        optimizer.zero_grad()
        for i, (node_feats, coords, global_feats, target) in enumerate(train_data):
            energy, _, uncertainty = model(node_feats, coords, global_feats)
            loss = criterion(energy.squeeze(), target.squeeze())
            (loss / accum_steps).backward()
            train_loss += loss.item()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / max(1, len(train_data))

        # Validation
        model.eval()
        val_errors = []
        with torch.no_grad():
            for node_feats, coords, global_feats, target in val_data:
                energy, _, _ = model(node_feats, coords, global_feats)
                err = abs(energy.squeeze().item() - target.squeeze().item())
                val_errors.append(err)

        val_mae = sum(val_errors) / max(1, len(val_errors))
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val MAE: {val_mae:.4f} eV | Time: {epoch_time:.1f}s", flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Save checkpoint
            save_payload = {
                "model_state": model.state_dict(),
                "val_mae": best_val_mae,
                "epochs": epoch,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(save_payload, OUTPUT_CHECKPOINT_PATH)
            torch.save(save_payload, CHECKPOINT_PATH)

    total_time = time.time() - start_time
    print("=" * 65)
    print(f"  CPU Fine-Tuning Complete in {total_time:.1f} seconds!")
    print(f"  -> Best Validation MAE: {best_val_mae:.4f} eV / atom")
    print(f"  -> Checkpoint Saved to: {OUTPUT_CHECKPOINT_PATH} and {CHECKPOINT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    run_cpu_finetuning(epochs=10, lr=1e-4)
