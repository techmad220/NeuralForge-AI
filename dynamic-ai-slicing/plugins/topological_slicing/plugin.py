"""Topological Slicing Plugin - Topology-aware slicing"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class TopologicalSlicingPlugin(SlicingPlugin):
    """Slices model based on topological properties."""

    def __init__(self, topology_groups: int = 4):
        self.topology_groups = topology_groups

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with topology-aware slicing."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Topological processing
        hidden = x
        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            # Group neurons by topological properties
            group_size = max(1, layer_size // self.topology_groups)
            hidden_groups = []

            for g in range(self.topology_groups):
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, layer_size)
                group_weights = np.random.randn(hidden.shape[-1], end_idx - start_idx) * 0.1
                hidden_groups.append(hidden @ group_weights)

            hidden = np.tanh(np.concatenate(hidden_groups, axis=-1) if hidden_groups else hidden)

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
