from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from .interfaces import SlicingPlugin, ContinualPlugin
from .plugin_loader import load_slicing_plugins, load_continual_plugins
from .chunked_model import load_metadata


class DynamicSlicingAI:
    """Facade around slicing + continual‑learning plugins."""

    def __init__(
        self,
        model_dir: str,
        slicing: Optional[str] = None,
        continual: Optional[str] = None,
    ) -> None:
        self.model_dir = model_dir
        self._slicing_plugins = load_slicing_plugins()
        self._continual_plugins = load_continual_plugins()
        self.slicing_name = slicing or next(iter(self._slicing_plugins.keys()), None)
        self.continual_name = continual or next(iter(self._continual_plugins.keys()), None)

        if self.slicing_name is None:
            raise RuntimeError("No slicing plugins found.")
        if self.continual_name is None:
            raise RuntimeError("No continual‑learning plugins found.")

        self.slicing: SlicingPlugin = self._slicing_plugins[self.slicing_name]()  # type: ignore[call‑arg]
        self.continual: ContinualPlugin = self._continual_plugins[self.continual_name]()  # type: ignore[call‑arg]
        self.meta = load_metadata(model_dir)

    def run_inference(self, x: np.ndarray, return_logits: bool = False) -> np.ndarray:
        if return_logits:
            return self.slicing.infer_logits(self.model_dir, x)
        return self.slicing.infer_proba(self.model_dir, x)

    def learn_from_other_ai(
        self,
        other_ai_model_dir: str,
        dataset: Iterable[np.ndarray],
        lambda_l2: float = 0.05,
        temperature: float = 2.0,
    ) -> None:
        self.continual.update_with_new_model(
            student_model_dir=self.model_dir,
            teacher_model_dir=other_ai_model_dir,
            dataset=dataset,
            lambda_l2=lambda_l2,
            temperature=temperature,
        )

    @staticmethod
    def available_plugins() -> dict:
        return {
            "slicing": list(load_slicing_plugins().keys()),
            "continual": list(load_continual_plugins().keys()),
        }