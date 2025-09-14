from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol
import numpy as np


class SlicingPlugin(ABC):
    """Interface for slicing/backing‑store strategy."""

    @abstractmethod
    def infer_logits(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Return raw logits for a single input vector."""

    def infer_proba(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        logits = self.infer_logits(model_dir, x)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    @abstractmethod
    def penultimate_features(self, model_dir: str, x: np.ndarray) -> np.ndarray:
        """Return features from last hidden layer (before head)."""


class ContinualPlugin(ABC):
    """Interface for continual learning strategies."""

    @abstractmethod
    def update_with_new_model(
        self,
        student_model_dir: str,
        teacher_model_dir: str,
        dataset: Iterable[np.ndarray],
        lambda_l2: float,
        temperature: float,
    ) -> None:
        """Update student in‑place based on teacher outputs."""