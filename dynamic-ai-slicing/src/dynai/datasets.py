from __future__ import annotations

from typing import Iterable, Iterator
import numpy as np


class SyntheticVectorDataset(Iterable[np.ndarray]):
    """Yields random input vectors for KD or evaluation."""

    def __init__(self, d_in: int, n: int, seed: int = 0) -> None:
        self.d_in = d_in
        self.n = n
        self.seed = seed

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            yield rng.normal(size=(self.d_in,)).astype(np.float32)