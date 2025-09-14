from __future__ import annotations

import numpy as np

from dynai.core import DynamicSlicingAI
from dynai.chunked_model import create_random_model
from dynai.datasets import SyntheticVectorDataset


def test_end_to_end(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    create_random_model(str(student), d_in=6, hidden=[8, 8], d_out=3, seed=1)
    create_random_model(str(teacher), d_in=6, hidden=[8, 8], d_out=3, seed=2)

    ai = DynamicSlicingAI(str(student))
    x = np.ones((6,), dtype=np.float32)
    p = ai.run_inference(x)
    assert p.shape == (3,)
    assert abs(p.sum() - 1.0) < 1e-5

    ds = SyntheticVectorDataset(d_in=6, n=64, seed=0)
    ai.learn_from_other_ai(str(teacher), ds, lambda_l2=0.1, temperature=2.0)

    p2 = ai.run_inference(x)
    assert p2.shape == (3,)