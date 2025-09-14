# Dynamic AI Slicing & Self‑Learning (Plugin Architecture)

Run huge models on modest hardware by chunking layers to disk and loading them on demand.  
Continuously learn from other AIs without full backprop via a plugin system.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pre-commit install
```

## Quickstart

```bash
# 1) Make a tiny chunked model
dynai init-model --outdir ./student_model --d-in 8 --hidden 16 16 --d-out 4 --seed 1

# 2) Make a "teacher" (same dims; different random weights)
dynai init-model --outdir ./teacher_model --d-in 8 --hidden 16 16 --d-out 4 --seed 999

# 3) List plugins
dynai list-plugins

# 4) Inference (sliced loading; outputs softmax probs)
dynai infer --model-dir ./student_model --input "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]"

# 5) Continual learn from teacher via KD on head (ridge-regularized)
dynai learn-from --student ./student_model --teacher ./teacher_model --samples 256 --seed 42 --lambda 0.05 --temp 2.0
```

## Design

- **Slicing plugin** (`dynai.slicing`): controls how layers are loaded and executed on demand.
- **Continual plugin** (`dynai.continual`): strategies to update the student safely (example: KD on head).
- **Chunked format**: directory with `metadata.json` and per-layer `.npy` files (weights/biases).

## Notes

- Pure NumPy; portable.
- KD‑head updates only the last layer using ridge regression to match teacher logits; avoids overfitting.