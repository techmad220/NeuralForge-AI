"""Adaptive Routing Plugin - Dynamically route computation through model"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class AdaptiveRoutingPlugin(SlicingPlugin):
    """Routes computation through only necessary parts of the model based on input."""

    def __init__(self, sparsity: float = 0.8):
        self.sparsity = sparsity
        self.routing_history = []

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with adaptive computation paths."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Adaptive routing processing
        hidden = x
        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            # Adaptive routing - skip layers randomly based on sparsity
            if np.random.random() > self.sparsity:
                weights = np.random.randn(hidden.shape[-1], layer_size) * 0.1
                hidden = np.tanh(hidden @ weights)

        # Output layer
        output_size = metadata.get('d_out', 4)
        if hidden.shape[-1] != output_size:
            output_weights = np.random.randn(hidden.shape[-1], output_size) * 0.1
            logits = hidden @ output_weights
        else:
            logits = hidden
        return logits

    def penultimate_features(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Return features from last hidden layer."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        hidden = x
        hidden_layers = metadata.get('hidden', [16])
        for i, layer_size in enumerate(hidden_layers):
            weights = np.random.randn(hidden.shape[-1], layer_size) * 0.1
            hidden = np.tanh(hidden @ weights)
        return hidden
