from __future__ import annotations

import numpy as np

from dynai.interfaces import SlicingPlugin
from dynai.chunked_model import forward_logits_streaming, penultimate_features_streaming


class WindowedSlicing(SlicingPlugin):
    """
    Minimal streaming slicer: loads one layer at a time (weights + bias) to compute forward.
    Uses NumPy memmap under the hood for low‑RAM operation.
    """

    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        return forward_logits_streaming(model_dir, x)

    def penultimate_features(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        return penultimate_features_streaming(model_dir, x)