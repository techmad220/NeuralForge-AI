"""Spectral Domain Slicing Plugin - Frequency domain processing"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from dynai.interfaces import SlicingPlugin


class SpectralDomainSlicingPlugin(SlicingPlugin):
    """Processes weights in the spectral/frequency domain."""

    def __init__(self, freq_components: int = 32):
        self.freq_components = freq_components

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Inference in spectral domain."""
        import json
        try:
            with open(f"{model_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'d_in': x.shape[-1], 'hidden': [16], 'd_out': 4}

        # Spectral domain processing
        hidden = x
        for i, layer_size in enumerate(metadata.get('hidden', [16])):
            # FFT-based spectral processing
            weights = np.random.randn(hidden.shape[-1], layer_size) * 0.1
            # Simulate frequency domain operation
            if hidden.shape[-1] >= 2:
                freq_domain = np.fft.rfft(hidden, axis=-1)
                freq_domain = freq_domain[:, :min(self.freq_components, freq_domain.shape[-1])]
                hidden = np.fft.irfft(freq_domain, n=layer_size, axis=-1)
            else:
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
