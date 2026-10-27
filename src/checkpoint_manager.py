import os
import json
import time
import pandas as pd
import torch

from src.physics_constants import (
    DATA_DIR, CHECKPOINT_PATH, MEMORY_VAULT_PATH, STATE_PATH, LOG_HISTORY_PATH
)

class CheckpointManager:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def save_all(self, total_steps, current_loss, discovered_vault, live_logs):
        try:
            # 1. PyTorch Model Checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps': total_steps,
                'current_loss': current_loss
            }, CHECKPOINT_PATH)

            # 2. Vault & CSV Outputs
            with open(MEMORY_VAULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(discovered_vault, f, indent=2)

            if discovered_vault:
                pd.DataFrame(discovered_vault).to_csv(
                    os.path.join(DATA_DIR, "05_mcrsa_feature_transformation_summary.csv"), index=False
                )

            # 3. Live Logs Storage
            with open(LOG_HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(live_logs[-200:], f, indent=2)

            # 4. State JSON
            last_time = time.strftime("%Y-%m-%d %H:%M:%S")
            state_data = {
                'total_steps': total_steps,
                'total_discoveries': len(discovered_vault),
                'current_loss': round(current_loss, 6),
                'last_checkpoint_time': last_time
            }
            with open(STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)

            return last_time
        except Exception as e:
            print(f"Error in CheckpointManager.save_all: {e}")
            return "Error"

    def load_all(self):
        total_steps = 0
        current_loss = 0.0
        discovered_vault = []
        live_logs = []
        last_time = "None"

        try:
            if os.path.exists(CHECKPOINT_PATH):
                ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
                state_dict = ckpt.get('model_state_dict') or ckpt.get('model_state') or ckpt
                self.model.load_state_dict(state_dict, strict=False)
                if 'optimizer_state_dict' in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception:
                        pass
                total_steps = ckpt.get('total_steps', 0)
                current_loss = ckpt.get('current_loss', 0.0)

            if os.path.exists(MEMORY_VAULT_PATH):
                with open(MEMORY_VAULT_PATH, 'r', encoding='utf-8') as f:
                    discovered_vault = json.load(f)

            if os.path.exists(LOG_HISTORY_PATH):
                with open(LOG_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    live_logs = json.load(f)

            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                    last_time = st.get('last_checkpoint_time', 'None')

        except Exception as e:
            print(f"Error in CheckpointManager.load_all: {e}")

        return total_steps, current_loss, discovered_vault, live_logs, last_time
