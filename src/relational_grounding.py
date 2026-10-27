import torch

from src.features import MaterialFeaturePipeline
from src.models import MCRSA_EGNN
from src.physics_constants import ELEMENT_PHYSICS

class RelationalGroundingSpace:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.physics = ELEMENT_PHYSICS
        self.feature_pipeline = MaterialFeaturePipeline()

    def extract_node_features(self, elements):
        return self.feature_pipeline.node_features(elements).to(self.device)
