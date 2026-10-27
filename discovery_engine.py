import os
import json
import time
import math
import random
import copy
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import load_local_env
from src.features import MaterialFeaturePipeline
from src.models import MCRSA_EGNN
from src.acquisition import AcquisitionFunction
from src.candidate_generation import CompositionGenerator
from src.memory import DiscoveryMemory
from src.validation import ProposalValidator
from src.data_ingestion import (
    LiveMaterialsProjectSync,
    MaterialsProjectClient,
    MaterialsProjectPipeline,
)

load_local_env()

# Paths configuration
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(DATA_DIR, "mcrsa_checkpoint.pt")
MEMORY_VAULT_PATH = os.path.join(DATA_DIR, "mcrsa_memory_vault.json")
STATE_PATH = os.path.join(DATA_DIR, "mcrsa_state.json")
FEEDBACK_PATH = os.path.join(DATA_DIR, "mcrsa_external_feedback.json")
INGESTION_AUDIT_PATH = os.path.join(DATA_DIR, "materials_project_ingestion.jsonl")
ACCEPTED_DATA_PATH = os.path.join(DATA_DIR, "materials_project_accepted.json")
MP_FALLBACK_PATH = ACCEPTED_DATA_PATH

from src.physics_constants import ELEMENT_PHYSICS, PRECURSORS_A, PRECURSORS_B



# --- SELF-EVOLVING DISCOVERY ENGINE ---
class AutonomousDiscoveryEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_pipeline = MaterialFeaturePipeline()
        self.model = MCRSA_EGNN(
            in_features=self.feature_pipeline.node_feature_dim,
            global_features=self.feature_pipeline.global_feature_dim,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.acquisition = AcquisitionFunction()
        self.validator = ProposalValidator()
        self.discovery_memory = DiscoveryMemory(FEEDBACK_PATH)
        self.data_pipeline = MaterialsProjectPipeline(
            INGESTION_AUDIT_PATH,
            ACCEPTED_DATA_PATH,
            set(ELEMENT_PHYSICS),
        )
        api_key = os.environ.get("MP_API_KEY", "").strip()
        self.live_sync = (
            LiveMaterialsProjectSync(
                MaterialsProjectClient(api_key, supported_elements=set(ELEMENT_PHYSICS)),
                self.ingest_materials_project_records,
                MP_FALLBACK_PATH,
                initial_batch=int(os.environ.get("MP_INITIAL_BATCH", "10")),
                min_batch=int(os.environ.get("MP_MIN_BATCH", "5")),
                max_batch=int(os.environ.get("MP_MAX_BATCH", "25")),
                interval_seconds=float(os.environ.get("MP_SYNC_INTERVAL", "300")),
            )
            if api_key else None
        )

        self.is_running = False
        self.worker_thread = None

        self.dataset = self._load_dataset()
        accepted_records = self.data_pipeline.accepted_records()
        existing_ids = {
            item.get("material_id") for item in self.dataset if item.get("material_id")
        }
        self.dataset.extend(
            item for item in accepted_records
            if not item.get("material_id") or item.get("material_id") not in existing_ids
        )
        self._refresh_dataset_splits()
        self.benchmark_evaluation_limit = 256
        self.known_compositions = set()
        for d in self.dataset:
            self.known_compositions.add(tuple(sorted(d['elements'])))

        self.search_space_A = [
            symbol for symbol, properties in ELEMENT_PHYSICS.items()
            if properties['valence'] > 0 and properties['radius'] >= 1.0
        ]
        self.search_space_B = [
            symbol for symbol, properties in ELEMENT_PHYSICS.items()
            if properties['valence'] > 0 and properties['radius'] < 1.0
        ]
        self.search_space_X = [
            symbol for symbol, properties in ELEMENT_PHYSICS.items()
            if properties['valence'] < 0
        ]

        self.discovered_vault = []
        self.candidate_attempts = {}
        self.candidate_cycle = 0
        self.candidate_pool = []
        self._build_candidate_pool()
        self.current_formula = "Preparing candidate search..."
        self.total_steps = 0
        self.current_loss = 0.0
        self.current_uncertainty = 0.0
        self.validation_mae = 0.0
        self.validation_rmse = 0.0
        self.validation_r2 = 0.0
        self.validation_bias = 0.0
        self.validation_accuracy = 0.0
        self.validation_accuracy_tolerance = 0.1
        self.validation_evaluated_at_step = 0
        self.training_log = []
        self.loss_history = []
        self.validation_targets = []
        self.validation_predictions = []
        self.last_loop_error = None
        self.best_candidate_energy = None
        self.promising_threshold = 0.6
        self.last_acquisition_score = None
        self.last_proposal_source = "dynamic_composition_generator"
        self.external_feedback_count = len(self.discovery_memory.load_feedback())
        self.feedback_training_updates = 0
        self.last_checkpoint_time = "None"
        self.last_training_time = None
        self.last_progress_time = None
        self.training_phase = "pretraining"
        self.pretraining_completed = False
        self.pretraining_epochs_completed = 0
        self.pretraining_epochs = max(1, int(os.environ.get("PRETRAIN_EPOCHS", "8")))
        self.pretraining_batch_size = max(
            1, int(os.environ.get("PRETRAIN_BATCH_SIZE", "32"))
        )
        self.best_validation_mae = float("inf")
        self.best_model_state = None
        self.best_optimizer_state = None
        self.online_updates_since_benchmark = 0
        self.online_update_gate_interval = max(
            1, int(os.environ.get("ONLINE_GATE_INTERVAL", "1"))
        )
        
        # Load persistent state if available
        self.load_checkpoint()
        self._build_candidate_pool()
        if not self.candidate_pool:
            self.current_formula = "Search space exhausted"

    def _build_candidate_pool(self):
        """Build a dynamic, geometry-filtered pool without duplicate formulas."""
        discovered = {item['formula'] for item in self.discovered_vault}
        generator = CompositionGenerator(
            ELEMENT_PHYSICS,
            self.compute_goldschmidt_tolerance,
            discovered,
        )
        proposals = generator.propose(
            self.search_space_A,
            self.search_space_B,
            self.search_space_X,
        )
        self.candidate_pool = [
            candidate for candidate in proposals
        ]

    def _next_candidate(self):
        """Return an untried candidate, then start a new evaluation cycle."""
        if not self.candidate_pool:
            self._build_candidate_pool()
        if not self.candidate_pool:
            self.current_formula = "Search space exhausted"
            self.last_proposal_source = "no_novel_candidates"
            return None

        pool_size = len(self.candidate_pool)
        window_size = min(32, pool_size)
        window_start = (self.candidate_cycle * window_size) % pool_size
        window = [
            self.candidate_pool[(window_start + index) % pool_size]
            for index in range(window_size)
        ]
        candidate = max(window, key=self._candidate_acquisition_score)
        formula = candidate.formula
        self.candidate_attempts[formula] = self.candidate_attempts.get(formula, 0) + 1
        self.candidate_cycle += 1
        return candidate

    def _candidate_acquisition_score(self, candidate):
        """Score a proposal with the current model before selecting it."""
        elements, coords = self.construct_candidate_coords(candidate)
        self.model.eval()
        with torch.no_grad():
            node_feats, global_feats = self.feature_pipeline.tensors(
                elements, coords, self.device
            )
            coordinates = torch.tensor(coords, dtype=torch.float32).to(self.device)
            energy, _, uncertainty = self.model(node_feats, coordinates, global_feats)
        formula = candidate.formula
        acquisition = self.acquisition.score(
            float(energy.item()),
            float(uncertainty.item()),
            formula,
            {item['formula'] for item in self.discovered_vault},
        )
        return acquisition.score - 0.02 * self.candidate_attempts.get(formula, 0)

    def _load_dataset(self):
        dataset_path = os.path.join(DATA_DIR, "perovskites_3d_dataset.json")
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _refresh_dataset_splits(self):
        self.train_dataset, self.validation_dataset, self.test_dataset = (
            self._split_dataset(self.dataset)
        )
        targets = np.asarray(
            [float(item["target_energy"]) for item in self.train_dataset],
            dtype=np.float64,
        )
        self.target_mean = float(targets.mean()) if len(targets) else 0.0
        self.target_std = float(targets.std()) if len(targets) else 1.0
        self.target_std = max(self.target_std, 1e-6)

    @staticmethod
    def _split_dataset(dataset):
        """Create a deterministic representative development split."""
        if not dataset:
            return [], [], []
        ordered = list(dataset)
        random.Random(42).shuffle(ordered)
        total = len(ordered)
        train_end = max(1, int(total * 0.8))
        validation_end = max(train_end + 1, int(total * 0.9))
        validation_end = min(validation_end, total)
        return (
            ordered[:train_end],
            ordered[train_end:validation_end],
            ordered[validation_end:],
        )

    def _normalize_target(self, value):
        return (float(value) - self.target_mean) / self.target_std

    def _denormalize_target(self, value):
        return float(value) * self.target_std + self.target_mean

    def ingest_materials_project_records(self, records):
        """Audit and validate records before admitting them to the training dataset."""
        results = self.data_pipeline.ingest(records)
        accepted = [result.record for result in results if result.status == "accepted"]
        self.training_log.append({
            'time': time.strftime("%H:%M:%S"),
            'step': self.total_steps,
            'message': (
                f"Materials Project batch · {len(records)} received · "
                f"{sum(result.status == 'accepted' for result in results)} accepted · "
                f"{sum(result.status == 'duplicate' for result in results)} duplicate · "
                f"{sum(result.status == 'rejected' for result in results)} rejected"
            ),
            'loss': round(self.current_loss, 5),
        })
        self.training_log = self.training_log[-60:]
        existing_ids = {
            item.get("material_id") for item in self.dataset if item.get("material_id")
        }
        for record in accepted:
            if record.get("material_id") and record["material_id"] in existing_ids:
                continue
            self.dataset.append(record)
            if record.get("material_id"):
                existing_ids.add(record["material_id"])
        if accepted:
            self._refresh_dataset_splits()
        return results

    def sync_materials_project_once(self):
        if not self.live_sync:
            raise RuntimeError("MP_API_KEY is not configured")
        return self.live_sync.sync_once()

    def compute_goldschmidt_tolerance(self, A, B, X):
        rA = ELEMENT_PHYSICS[A]['radius']
        rB = ELEMENT_PHYSICS[B]['radius']
        rX = ELEMENT_PHYSICS[X]['radius']
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        return t, mu

    def construct_unit_cell_coords(self, A, B, X):
        coords = [
            [0.0, 0.0, 0.0],
            [1.95, 1.95, 1.95],
            [1.95, 1.95, 0.0],
            [1.95, 0.0, 1.95],
            [0.0, 1.95, 1.95]
        ]
        elements = [A, B, X, X, X]
        return elements, coords

    def construct_candidate_coords(self, candidate):
        if not candidate.mixed_a and not candidate.mixed_b:
            return self.construct_unit_cell_coords(
                candidate.elem_a, candidate.elem_b, candidate.elem_x
            )
        if candidate.mixed_a:
            elements = [
                candidate.elem_a, candidate.mixed_a, candidate.elem_b,
                candidate.elem_x, candidate.elem_x, candidate.elem_x,
            ]
        else:
            elements = [
                candidate.elem_a, candidate.elem_b, candidate.mixed_b,
                candidate.elem_x, candidate.elem_x, candidate.elem_x,
            ]
        coords = [
            [0.0, 0.0, 0.0],
            [1.95, 1.95, 1.95],
            [1.95, 1.95, 0.0],
            [1.95, 0.0, 1.95],
            [0.0, 1.95, 1.95],
            [0.975, 0.975, 0.975],
        ]
        return elements, coords

    def extract_node_features(self, elements):
        return self.feature_pipeline.node_features(elements).to(self.device)

    def train_step_on_structure(self, elements, coords, target_energy):
        self.model.train()
        self.optimizer.zero_grad()
        
        node_feats, global_feats = self.feature_pipeline.tensors(
            elements, coords, self.device
        )
        target_tensor = torch.tensor(
            [self._normalize_target(target_energy)], dtype=torch.float32
        ).to(self.device)
        
        x_coords = torch.tensor(coords, dtype=torch.float32).to(self.device)
        pred_energy, latent_z, _ = self.model(node_feats, x_coords, global_feats)
        loss = self.criterion(pred_energy, target_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return (
            loss.item(),
            self._denormalize_target(pred_energy.item()),
            latent_z.detach().cpu().numpy(),
        )

    def start_loop(self):
        if not self.is_running:
            if self.pretraining_completed and self.best_model_state is None:
                self._evaluate_benchmark()
                self.best_validation_mae = self.validation_mae
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state = copy.deepcopy(
                    self.optimizer.state_dict()
                )
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._autonomous_discovery_loop, daemon=True)
            self.worker_thread.start()
            print("Autonomous Discovery Loop started.")

    def pause_loop(self):
        if self.is_running:
            self.is_running = False
            if self.live_sync:
                self.live_sync.stop()
            if self.worker_thread and self.worker_thread is not threading.current_thread():
                self.worker_thread.join(timeout=2.0)
            self.save_checkpoint()
            print("Autonomous Discovery Loop paused & checkpoint saved.")

    def _run_pretraining(self):
        """Train on the complete admitted dataset before enabling discovery."""
        if self.pretraining_completed:
            return True
        if not self.dataset:
            self.last_loop_error = "Pretraining requires at least one dataset record"
            return False

        self.training_phase = "pretraining"
        self.current_formula = "Pretraining model on admitted dataset"
        train_source = self.train_dataset or self.dataset
        best_model_state = None
        best_optimizer_state = None
        for epoch in range(self.pretraining_epochs_completed, self.pretraining_epochs):
            if not self.is_running:
                return False
            indices = list(range(len(train_source)))
            random.shuffle(indices)
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(indices), self.pretraining_batch_size):
                if not self.is_running:
                    return False
                batch = [
                    train_source[index]
                    for index in indices[start:start + self.pretraining_batch_size]
                ]
                self.model.train()
                self.optimizer.zero_grad()
                batch_loss = torch.zeros((), device=self.device)
                for sample in batch:
                    node_feats, global_feats = self.feature_pipeline.tensors(
                        sample["elements"], sample["coordinates_3d"], self.device
                    )
                    coords = torch.tensor(
                        sample["coordinates_3d"], dtype=torch.float32, device=self.device
                    )
                    target = torch.tensor(
                        [self._normalize_target(sample["target_energy"])],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    prediction, _, _ = self.model(node_feats, coords, global_feats)
                    batch_loss = batch_loss + self.criterion(prediction, target)
                batch_loss = batch_loss / max(len(batch), 1)
                batch_loss.backward()
                self.optimizer.step()

                loss_value = float(batch_loss.item())
                epoch_loss += loss_value
                batch_count += 1
                self.total_steps += 1
                self.current_loss = (
                    0.95 * self.current_loss + 0.05 * loss_value
                    if self.current_loss > 0 else loss_value
                )
                self.loss_history.append(round(self.current_loss, 6))
                self.loss_history = self.loss_history[-80:]
                self.last_training_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.last_progress_time = time.monotonic()

            self.pretraining_epochs_completed = epoch + 1
            self._evaluate_benchmark()
            if self.validation_mae < self.best_validation_mae:
                self.best_validation_mae = self.validation_mae
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            self.training_log.append({
                "time": time.strftime("%H:%M:%S"),
                "step": self.total_steps,
                "message": (
                    f"pretraining epoch {self.pretraining_epochs_completed}/"
                    f"{self.pretraining_epochs} complete · "
                    f"{len(train_source)} records · batch {self.pretraining_batch_size}"
                ),
                "loss": round(epoch_loss / max(batch_count, 1), 5),
            })
            self.training_log = self.training_log[-60:]
            self.save_checkpoint()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if best_optimizer_state is not None:
                self.optimizer.load_state_dict(best_optimizer_state)
            self._evaluate_benchmark()
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            self.best_validation_mae = self.validation_mae
        self.pretraining_completed = True
        self.training_phase = "discovery"
        self.current_formula = "Starting candidate discovery"
        self.save_checkpoint()
        if self.live_sync and not self.live_sync.is_running:
            self.live_sync.start()
        return True

    def _autonomous_discovery_loop(self):
        dataset_idx = 0
        while self.is_running:
            try:
                if not self.pretraining_completed:
                    if not self._run_pretraining():
                        time.sleep(0.5)
                    continue
                self.training_phase = "discovery"
                if len(self.dataset) > 0:
                    train_source = self.train_dataset or self.dataset
                    sample = train_source[dataset_idx % len(train_source)]
                    dataset_idx += 1
                    loss_val, pred_energy, _ = self.train_step_on_structure(
                        sample['elements'], sample['coordinates_3d'], sample['target_energy']
                    )
                    self.total_steps += 1
                    self.last_training_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.last_progress_time = time.monotonic()
                    self.current_loss = 0.9 * self.current_loss + 0.1 * loss_val if self.current_loss > 0 else loss_val
                    self.loss_history.append(round(self.current_loss, 6))
                    self.loss_history = self.loss_history[-80:]
                    self.online_updates_since_benchmark += 1
                    if (
                        self.online_updates_since_benchmark
                        >= self.online_update_gate_interval
                    ):
                        self._evaluate_benchmark()
                        if (
                            self.best_model_state is not None
                            and self.validation_mae > self.best_validation_mae + 0.01
                        ):
                            self.model.load_state_dict(self.best_model_state)
                            if self.best_optimizer_state is not None:
                                self.optimizer.load_state_dict(
                                    self.best_optimizer_state
                                )
                            self._evaluate_benchmark()
                            self.training_log.append({
                                'time': time.strftime("%H:%M:%S"),
                                'step': self.total_steps,
                                'message': (
                                    'online update rejected · validation regression '
                                    f'({self.validation_mae:.4f} > '
                                    f'{self.best_validation_mae:.4f})'
                                ),
                                'loss': round(loss_val, 5),
                            })
                        elif self.validation_mae < self.best_validation_mae:
                            self.best_validation_mae = self.validation_mae
                            self.best_model_state = copy.deepcopy(
                                self.model.state_dict()
                            )
                            self.best_optimizer_state = copy.deepcopy(
                                self.optimizer.state_dict()
                            )
                        self.online_updates_since_benchmark = 0
                    self.training_log.append({
                        'time': time.strftime("%H:%M:%S"),
                        'step': self.total_steps,
                        'message': f"gradient update · {sample['elements'][0]}{sample['elements'][1]}{sample['elements'][2]}3",
                        'loss': round(loss_val, 5),
                    })
                    self.training_log = self.training_log[-60:]

                candidate = self._next_candidate()
                if candidate is None:
                    self.last_proposal_source = "no_novel_candidates"
                    time.sleep(0.3)
                    continue
                formula = candidate.formula
                self.current_formula = formula
                self.last_proposal_source = "dynamic_acquisition"

                A, B, X = candidate.elem_a, candidate.elem_b, candidate.elem_x
                comp_tuple = tuple(sorted(
                    self.construct_candidate_coords(candidate)[0]
                ))

                t_factor, mu_factor = self.compute_goldschmidt_tolerance(A, B, X)
                elems, coords = self.construct_candidate_coords(candidate)

                self.model.eval()
                with torch.no_grad():
                    node_feats, global_feats = self.feature_pipeline.tensors(
                        elems, coords, self.device
                    )
                    x_coords = torch.tensor(coords, dtype=torch.float32).to(self.device)
                    pred_energy_tensor, latent_z_tensor, uncertainty_tensor = self.model(
                        node_feats, x_coords, global_feats
                    )
                    est_Ef = self._denormalize_target(pred_energy_tensor.item())
                    latent_z = latent_z_tensor.cpu().numpy().tolist()
                    uncertainty = float(uncertainty_tensor.item())
                    self.current_uncertainty = uncertainty

                acquisition = self.acquisition.score(
                    est_Ef,
                    uncertainty,
                    formula,
                    {item['formula'] for item in self.discovered_vault},
                )
                self.last_acquisition_score = round(acquisition.score, 5)
                proposal_validation = self.validator.classify(t_factor, est_Ef)
                if self.best_candidate_energy is None or est_Ef < self.best_candidate_energy:
                    self.best_candidate_energy = est_Ef
                # Keep the original physical cutoff visible, but continue
                # exploring when model calibration is shifted above it.
                self.promising_threshold = max(0.6, self.best_candidate_energy + 0.1)
                is_physically_stable = proposal_validation.physically_stable
                is_model_promising = proposal_validation.status != 'rejected'

                if is_model_promising and not any(item['formula'] == formula for item in self.discovered_vault):
                    prec_A = PRECURSORS_A.get(A, f"{A} Oxide")
                    prec_B = PRECURSORS_B.get(B, f"{B} Oxide")
                    calc_temp = int(900 + 200 * t_factor + 50 * max(0, est_Ef))
                    reaction = f"{prec_A} + {prec_B} --({calc_temp}°C)--> {formula} + CO2↑"

                    discovery_entry = {
                        'step': self.total_steps,
                        'formula': formula,
                        'elem_A': A,
                        'elem_B': B,
                        'elem_X': X,
                        'elements': elems,
                        'coordinates_3d': coords,
                        'tolerance_t': round(t_factor, 4),
                        'octahedral_mu': round(mu_factor, 4),
                        'estimated_Ef_eV': round(est_Ef, 4),
                        'prediction_uncertainty_eV': round(uncertainty, 4),
                        'is_physically_stable': is_physically_stable,
                        'discovery_status': (
                            proposal_validation.status
                        ),
                        'acquisition_score': self.last_acquisition_score,
                        'validation_source': 'model_prediction_only',
                        'requires_external_validation': proposal_validation.requires_external_validation,
                        'precursor_A': prec_A,
                        'precursor_B': prec_B,
                        'calcination_temp_C': calc_temp,
                        'reaction': reaction,
                        'latent_z_dim': len(latent_z),
                        'discovered_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    self.discovered_vault.insert(0, discovery_entry)
                    self.known_compositions.add(comp_tuple)

                    if len(self.discovered_vault) % 5 == 0:
                        self.save_checkpoint()
                    self._build_candidate_pool()

                self.last_loop_error = None
                time.sleep(0.3)
            except Exception as error:
                self.last_loop_error = f"{type(error).__name__}: {error}"
                self.training_log.append({
                    'time': time.strftime("%H:%M:%S"),
                    'step': self.total_steps,
                    'message': f"engine error · {self.last_loop_error}",
                    'loss': round(self.current_loss, 5),
                })
                self.training_log = self.training_log[-60:]
                print(f"Discovery loop error: {self.last_loop_error}")
                time.sleep(1.0)

    def save_checkpoint(self):
        try:
            torch.save({
                'model_version': 2,
                'node_feature_dim': self.feature_pipeline.node_feature_dim,
                'global_feature_dim': self.feature_pipeline.global_feature_dim,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'current_loss': self.current_loss
                ,
                'pretraining_completed': self.pretraining_completed,
                'pretraining_epochs_completed': self.pretraining_epochs_completed,
            }, CHECKPOINT_PATH)
            
            with open(MEMORY_VAULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.discovered_vault, f, indent=2)

            if self.discovered_vault:
                pd.DataFrame(self.discovered_vault).to_csv(
                    os.path.join(DATA_DIR, "05_mcrsa_feature_transformation_summary.csv"), index=False
                )

            self.last_checkpoint_time = time.strftime("%Y-%m-%d %H:%M:%S")
            state_data = {
                'total_steps': self.total_steps,
                'total_discoveries': len(self.discovered_vault),
                'current_loss': round(self.current_loss, 6),
                'candidate_attempts': self.candidate_attempts,
                'candidate_cycle': self.candidate_cycle,
                'last_checkpoint_time': self.last_checkpoint_time
                ,
                'pretraining_completed': self.pretraining_completed,
                'pretraining_epochs_completed': self.pretraining_epochs_completed,
            }
            with open(STATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)

            print(f"Checkpoint successfully saved at {self.last_checkpoint_time}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def record_external_feedback(self, formula, observed_energy, source):
        """Persist an external label without treating model predictions as truth."""
        if not formula or not isinstance(observed_energy, (int, float)):
            raise ValueError("formula and numeric observed_energy are required")
        record = {
            'formula': formula,
            'observed_energy_eV': float(observed_energy),
            'source': source or 'external',
            'recorded_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.discovery_memory.append_feedback(record)
        self.external_feedback_count += 1
        matching = next(
            (item for item in self.discovered_vault if item.get('formula') == formula),
            None,
        )
        if matching and matching.get('coordinates_3d') and matching.get('elements'):
            loss, _, _ = self.train_step_on_structure(
                matching['elements'],
                matching['coordinates_3d'],
                float(observed_energy),
            )
            self.current_loss = loss
            self.feedback_training_updates += 1
        return record

    def load_checkpoint(self):
        try:
            if os.path.exists(CHECKPOINT_PATH):
                ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
                state = ckpt.get('model_state_dict') or ckpt.get('model_state') or ckpt
                try:
                    self.model.load_state_dict(state, strict=False)
                    if 'optimizer_state_dict' in ckpt:
                        try:
                            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        except Exception:
                            pass
                    model_loaded = True
                except RuntimeError as error:

                    print(
                        "Legacy checkpoint architecture detected; starting the upgraded "
                        f"model with fresh weights: {error}"
                    )
                    self.model = MCRSA_EGNN(
                        in_features=self.feature_pipeline.node_feature_dim,
                        global_features=self.feature_pipeline.global_feature_dim,
                    ).to(self.device)
                    self.optimizer = optim.AdamW(
                        self.model.parameters(), lr=1e-3, weight_decay=1e-4
                    )
                self.total_steps = ckpt.get('total_steps', 0)
                self.current_loss = ckpt.get('current_loss', 0.0) if model_loaded else 0.0
                if model_loaded:
                    print(
                        f"Restored PyTorch Checkpoint from {CHECKPOINT_PATH} "
                        f"(Steps: {self.total_steps})"
                    )
                else:
                    print(
                        f"Restored training counter from {CHECKPOINT_PATH}; "
                        "model weights were reset for the upgraded architecture."
                    )

            if os.path.exists(MEMORY_VAULT_PATH):
                with open(MEMORY_VAULT_PATH, 'r', encoding='utf-8') as f:
                    self.discovered_vault = json.load(f)
                    for item in self.discovered_vault:
                        if 'elements' in item:
                            self.known_compositions.add(tuple(sorted(item['elements'])))
                print(f"Restored Memory Vault with {len(self.discovered_vault)} discovered materials.")

            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                    self.candidate_attempts = {
                        str(formula): int(attempts)
                        for formula, attempts in st.get('candidate_attempts', {}).items()
                    }
                    self.candidate_cycle = int(st.get('candidate_cycle', 0))
                    self.last_checkpoint_time = st.get('last_checkpoint_time', 'None')
                    self.pretraining_completed = bool(
                        st.get('pretraining_completed', False)
                    )
                    self.pretraining_epochs_completed = int(
                        st.get('pretraining_epochs_completed', 0)
                    )
                    self.training_phase = (
                        "discovery" if self.pretraining_completed else "pretraining"
                    )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def get_neural_graph_data(self):
        nodes = []
        edges = []
        
        feature_names = ['Ionic Radius', 'Electronegativity', 'Valence State', 'Atomic Mass']
        for i, name in enumerate(feature_names):
            nodes.append({'id': f"in_{i}", 'label': name, 'group': 'input', 'x': -2.0, 'y': i*0.8 - 1.2, 'z': 0.0})
            
        for i in range(16):
            angle = (2 * math.pi * i) / 16
            nodes.append({
                'id': f"hid_{i}", 'label': f"Neuron_{i}", 'group': 'hidden',
                'x': 0.0, 'y': 1.5 * math.cos(angle), 'z': 1.5 * math.sin(angle),
                'activity': float(torch.sigmoid(self.model.blocks[i % len(self.model.blocks)].norm.weight.mean()).item())
            })

        nodes.append({'id': 'out_ef', 'label': 'Formation Energy ΔEf', 'group': 'output', 'x': 2.0, 'y': 0.8, 'z': 0.0})
        nodes.append({'id': 'out_t',  'label': 'Goldschmidt Tolerance t', 'group': 'output', 'x': 2.0, 'y': -0.8, 'z': 0.0})

        for i in range(len(feature_names)):
            for j in range(0, 16, 2):
                weight = float(self.model.node_proj[0].weight[:, i % self.model.in_features].abs().mean().item())
                edges.append({'source': f"in_{i}", 'target': f"hid_{j}", 'weight': weight})
        for j in range(16):
            weight = float(self.model.attention_pool[0].weight[:, j % self.model.attention_pool[0].in_features].abs().mean().item())
            edges.append({'source': f"hid_{j}", 'target': 'out_ef', 'weight': weight})
            edges.append({'source': f"hid_{j}", 'target': 'out_t', 'weight': weight * 0.8})

        return {'nodes': nodes, 'edges': edges}

    def get_telemetry(self):
        weight_values = torch.cat([parameter.detach().abs().flatten() for parameter in self.model.parameters()])
        targets = np.asarray(self.validation_targets, dtype=np.float64)
        predictions = np.asarray(self.validation_predictions, dtype=np.float64)
        if len(targets) > 1:
            residuals = predictions - targets
            total_variance = float(np.sum((targets - targets.mean()) ** 2))
            r2 = 1.0 - float(np.sum(residuals ** 2)) / max(total_variance, 1e-8)
            bias = float(residuals.mean())
        else:
            r2 = 0.0
            bias = 0.0
        return {
            'loss_history': self.loss_history,
            'logs': self.training_log,
            'metrics': {
            'training_phase': self.training_phase,
            'pretraining_completed': self.pretraining_completed,
            'pretraining_epochs_completed': self.pretraining_epochs_completed,
            'pretraining_epochs': self.pretraining_epochs,
            'training_records': len(self.train_dataset),
            'target_mean_eV': round(self.target_mean, 5),
            'target_std_eV': round(self.target_std, 5),
            'mae': round(self.validation_mae, 5),
                'rmse': round(self.validation_rmse, 5),
                'uncertainty': round(self.current_uncertainty, 5),
                'r2': round(self.validation_r2 if targets.size else r2, 5),
                'bias': round(self.validation_bias if targets.size else bias, 5),
                'accuracy': round(self.validation_accuracy, 5),
                'accuracy_percent': round(self.validation_accuracy * 100, 2),
                'accuracy_tolerance_eV': self.validation_accuracy_tolerance,
                'validation_samples': len(self.validation_targets),
                'benchmark_status': (
                    'ready' if self.validation_targets else 'pending_first_epoch'
                ),
                'test_samples': len(self.test_dataset),
                'benchmark_split': {
                    'train': len(self.train_dataset),
                    'validation': len(self.validation_dataset),
                    'test': len(self.test_dataset),
                },
                'benchmark_evaluated_at_step': self.validation_evaluated_at_step,
                'candidate_pool': len(self.candidate_pool),
                'candidate_pool_exhausted': len(self.candidate_pool) == 0,
                'search_status': (
                    'pretraining' if not self.pretraining_completed
                    else 'exhausted' if not self.candidate_pool else 'active'
                ),
                'candidate_cycle': self.candidate_cycle,
                'candidate_attempts': sum(self.candidate_attempts.values()),
                'strict_discoveries': sum(
                    item.get('is_physically_stable', False)
                    for item in self.discovered_vault
                ),
                'promising_candidates': sum(
                    item.get('discovery_status') == 'model_ranked_requires_validation'
                    for item in self.discovered_vault
                ),
                'promising_threshold': round(self.promising_threshold, 5),
                'best_candidate_energy': (
                    round(self.best_candidate_energy, 5)
                    if self.best_candidate_energy is not None else None
                ),
                'last_loop_error': self.last_loop_error,
                'last_acquisition_score': self.last_acquisition_score,
                'proposal_source': self.last_proposal_source,
                'external_feedback_count': self.external_feedback_count,
                'feedback_training_updates': self.feedback_training_updates,
                'worker_thread_alive': bool(
                    self.worker_thread and self.worker_thread.is_alive()
                ),
                'last_training_time': self.last_training_time,
                'seconds_since_progress': (
                    round(time.monotonic() - self.last_progress_time, 2)
                    if self.last_progress_time is not None else None
                ),
                'ingestion': self.data_pipeline.stats(),
                'materials_project_sync': (
                    self.live_sync.status()
                    if self.live_sync else {
                        'status': 'disabled',
                        'reason': 'MP_API_KEY is not configured',
                    }
                ),
                'weight_mean': round(float(weight_values.mean().item()), 5),
                'weight_std': round(float(weight_values.std().item()), 5),
                'parameters': sum(parameter.numel() for parameter in self.model.parameters()),
            },
        }

    def _evaluate_benchmark(self):
        """Evaluate the frozen validation split without updating model weights."""
        evaluation_set = (self.validation_dataset or self.test_dataset)[
            : self.benchmark_evaluation_limit
        ]
        if not evaluation_set:
            return
        self.model.eval()
        targets = []
        predictions = []
        with torch.no_grad():
            for sample in evaluation_set:
                node_feats, global_feats = self.feature_pipeline.tensors(
                    sample['elements'], sample['coordinates_3d'], self.device
                )
                coords = torch.tensor(
                    sample['coordinates_3d'], dtype=torch.float32
                ).to(self.device)
                prediction, _, _ = self.model(node_feats, coords, global_feats)
                targets.append(float(sample['target_energy']))
                predictions.append(self._denormalize_target(prediction.item()))
        self.validation_targets = targets
        self.validation_predictions = predictions
        residuals = np.asarray(predictions) - np.asarray(targets)
        self.validation_mae = float(np.mean(np.abs(residuals)))
        self.validation_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        self.validation_bias = float(np.mean(residuals))
        total_variance = float(np.sum((np.asarray(targets) - np.mean(targets)) ** 2))
        self.validation_r2 = (
            1.0 - float(np.sum(residuals ** 2)) / max(total_variance, 1e-8)
            if len(targets) > 1 else 0.0
        )
        self.validation_accuracy = float(
            np.mean(np.abs(residuals) <= self.validation_accuracy_tolerance)
        )
        self.validation_evaluated_at_step = self.total_steps

# Global Engine Instance
engine = AutonomousDiscoveryEngine()
