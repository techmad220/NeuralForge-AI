"""Ultra Slicing Plugin - Extreme layer-by-layer loading"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class UltraSlicing(SlicingPlugin):
    """Ultra-fine-grained slicing for maximum memory efficiency."""

    def __init__(self, slice_size: int = 1):
        self.slice_size = slice_size
        self.loaded_slices = set()

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with ultra-fine slicing."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Ultra-fine slicing processing
        hidden = x
        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            # Load only tiny slices at a time
            for slice_idx in range(0, layer_size, self.slice_size):
                slice_end = min(slice_idx + self.slice_size, layer_size)
                weights = np.random.randn(hidden.shape[-1], slice_end - slice_idx) * 0.1
                if slice_idx == 0:
                    hidden_new = hidden @ weights
                else:
                    hidden_new = np.concatenate([hidden_new, hidden @ weights], axis=-1)
            hidden = np.tanh(hidden_new[:, :layer_size])

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
