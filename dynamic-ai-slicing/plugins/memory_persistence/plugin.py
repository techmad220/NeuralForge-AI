"""Memory Persistence Plugin - Long-term memory across inferences"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class MemoryPersistencePlugin(SlicingPlugin):
    """Maintains persistent memory across multiple inference calls."""

    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.persistent_memory = []

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with persistent memory."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Memory persistence processing
        hidden = x
        # Use persistent memory
        if len(self.persistent_memory) > 0:
            memory_context = np.mean(self.persistent_memory[-10:], axis=0)
            hidden = (hidden + memory_context) / 2

        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            weights = np.random.randn(hidden.shape[-1], layer_size) * 0.1
            hidden = np.tanh(hidden @ weights)

        # Store in memory
        if len(self.persistent_memory) < self.memory_size:
            self.persistent_memory.append(hidden.copy())

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
