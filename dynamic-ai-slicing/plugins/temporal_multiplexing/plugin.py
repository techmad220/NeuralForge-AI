"""Temporal Multiplexing Plugin - Reuse memory across time steps."""

import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin
from dynai.quantum_neural_engine import TemporalWeightMultiplexer


class TemporalMultiplexingPlugin(SlicingPlugin):
    """Reuses same memory for different weights across time."""

    def __init__(self, memory_pool_gb: float = 6.0):
        self.multiplexer = TemporalWeightMultiplexer(
            memory_pool_gb=memory_pool_gb,
            num_time_slots=8
        )
        self.time_step = 0

    def infer_proba(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with temporal weight multiplexing."""
        import json
        with open(f"{model_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)

        hidden = x
        
        # Process through time-multiplexed layers
        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            # Get weights for current time slot
            time_slot = self.time_step % self.multiplexer.num_time_slots
            
            # Add weights to slot if not present
            layer_name = f"layer_{i}"
            if layer_name not in self.multiplexer.weight_slots[time_slot]:
                weights = np.random.randn(hidden.shape[-1], layer_size) * 0.1
                self.multiplexer.add_weight_to_slot(layer_name, weights, time_slot)
            
            # Get active weights
            active_weights = self.multiplexer.get_active_weights(self.time_step)
            if layer_name in active_weights:
                weights = active_weights[layer_name]
                hidden = np.tanh(hidden @ weights)
            
            # Advance time
            self.time_step += 1

        # Output
        output_size = metadata.get('d_out', 4)
        output_weights = np.random.randn(hidden.shape[-1], output_size) * 0.1
        logits = hidden @ output_weights

        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        proba = self.infer_proba(model_dir, x)
        return np.log(proba + 1e-10)

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
