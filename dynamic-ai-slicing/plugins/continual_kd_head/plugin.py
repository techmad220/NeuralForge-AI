from __future__ import annotations

from typing import Iterable
import numpy as np

from dynai.interfaces import ContinualPlugin
from dynai.chunked_model import (
    load_metadata,
    penultimate_features_streaming,
    forward_logits_streaming,
    overwrite_head_weights,
)


class KDHeadContinual(ContinualPlugin):
    """
    Knowledge Distillation on the final layer (head) only.
    - Collect student penultimate features F (NxH)
    - Collect teacher logits Z (NxC), scaled by temperature for stability
    - Solve ridge regression: minimize ||F*W + b - Z||^2 + λ||W||^2
      Closed‑form: W' = (F'^T F' + λI)^‑1 F'^T Z, where F' = [F | 1]
    - Overwrite student's head weights with W and b.
    """

    def update_with_new_model(
        self,
        student_model_dir: str,
        teacher_model_dir: str,
        dataset: Iterable[np.ndarray],
        lambda_l2: float,
        temperature: float,
    ) -> None:
        sm = load_metadata(student_model_dir)
        tm = load_metadata(teacher_model_dir)

        # Basic shape checks
        if (sm["d_in"], sm["hidden"], sm["d_out"]) != (tm["d_in"], tm["hidden"], tm["d_out"]):
            raise ValueError("Student and teacher architectures must match for KD‑head.")

        # Collect features and teacher logits
        feats = []
        t_logits = []
        for x in dataset:
            f = penultimate_features_streaming(student_model_dir, x)
            zt = forward_logits_streaming(teacher_model_dir, x)
            feats.append(f)
            # scale teacher logits by temperature
            t_logits.append(zt / max(temperature, 1e‑6))

        F = np.stack(feats, axis=0)  # (N, H)
        Z = np.stack(t_logits, axis=0)  # (N, C)

        # Augment F with bias term
        N, H = F.shape
        _, C = Z.shape
        F1 = np.concatenate([F, np.ones((N, 1), dtype=F.dtype)], axis=1)  # (N, H+1)

        reg = lambda_l2 * np.eye(H + 1, dtype=F.dtype)
        A = F1.T @ F1 + reg  # (H+1, H+1)
        B = F1.T @ Z         # (H+1, C)

        # Solve for W' (H+1, C)
        W_full = np.linalg.solve(A, B)  # stable for small H; for big H use lstsq
        W = W_full[:H, :].astype(np.float32)
        b = W_full[H, :].astype(np.float32)

        # Persist to student's head
        overwrite_head_weights(student_model_dir, W=W, b=b)