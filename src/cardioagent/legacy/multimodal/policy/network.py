"""Small action-value network over numeric evidence."""

import numpy as np
import torch
from torch import nn

from cardioagent.legacy.multimodal.policy.value import OUTPUT_SLOTS


class ValueNetwork(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.layers = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, len(OUTPUT_SLOTS)))

    def forward(self, values):
        return torch.sigmoid(self.layers(values))

    def predict(self, values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self(torch.from_numpy(values)).numpy()
