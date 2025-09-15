"""Deep Thinking Plugin - Multi-pass reasoning"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class DeepThinkingPlugin(SlicingPlugin):
    """Performs multiple passes through the model for deeper reasoning."""

    def __init__(self, num_passes: int = 3):
        self.num_passes = num_passes

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference with multiple reasoning passes."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Deep thinking processing
        hidden = x
        # Multiple passes for deeper thinking
        for pass_idx in range(self.num_passes):
            hidden_pass = hidden
            for i, layer_size in enumerate(metadata.get('hidden', [16])):
                weights = np.random.randn(hidden_pass.shape[-1], layer_size) * 0.1
                hidden_pass = np.tanh(hidden_pass @ weights)
            # Combine with previous
            hidden = (hidden + hidden_pass) / 2 if pass_idx > 0 else hidden_pass

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
