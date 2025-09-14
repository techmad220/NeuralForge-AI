from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _layer_paths(base: Path, idx: int) -> Tuple[Path, Path]:
    return base / f"layer{idx}_w.npy", base / f"layer{idx}_b.npy"


def create_random_model(
    outdir: str,
    d_in: int,
    hidden: List[int],
    d_out: int,
    seed: int = 1,
) -> None:
    """Create a random MLP split into per‑layer .npy files."""
    rng = np.random.default_rng(seed)
    dims = [d_in] + hidden + [d_out]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(len(dims) - 1):
        w = rng.standard_normal((dims[i], dims[i + 1]), dtype=np.float32) * (1.0 / np.sqrt(dims[i]))
        b = np.zeros((dims[i + 1],), dtype=np.float32)
        wp, bp = _layer_paths(out, i)
        np.save(wp, w)
        np.save(bp, b)

    meta = {
        "d_in": d_in,
        "hidden": hidden,
        "d_out": d_out,
        "layers": len(dims) - 1,
        "activation": "relu",
        "output": "logits",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_metadata(model_dir: str) -> Dict:
    return json.loads((Path(model_dir) / "metadata.json").read_text())


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z, dtype=np.float64)
    return (e / e.sum()).astype(x.dtype)


def forward_logits_streaming(model_dir: str, x: np.ndarray) -> np.ndarray:
    """Compute logits layer‑by‑layer, loading weights on demand."""
    meta = load_metadata(model_dir)
    layers = meta["layers"]
    h = x.astype(np.float32)

    for i in range(layers):
        wp, bp = _layer_paths(Path(model_dir), i)
        W = np.load(wp, mmap_mode="r")
        b = np.load(bp, mmap_mode="r")
        h = (h @ W) + b
        if i < layers - 1:
            h = relu(h)

        # release memmap references early (hint to GC)
        del W
        del b
    return h  # logits


def penultimate_features_streaming(model_dir: str, x: np.ndarray) -> np.ndarray:
    """Return the features from last hidden layer (before head)."""
    meta = load_metadata(model_dir)
    layers = meta["layers"]
    h = x.astype(np.float32)
    for i in range(layers - 1):  # stop before head
        wp, bp = _layer_paths(Path(model_dir), i)
        W = np.load(wp, mmap_mode="r")
        b = np.load(bp, mmap_mode="r")
        h = (h @ W) + b
        if i < layers - 1:
            h = relu(h)
        del W
        del b
    return h


def overwrite_head_weights(model_dir: str, W: np.ndarray, b: np.ndarray) -> None:
    """Persist new head weights to disk."""
    meta = load_metadata(model_dir)
    head_idx = meta["layers"] - 1
    wp, bp = _layer_paths(Path(model_dir), head_idx)
    np.save(wp, W.astype(np.float32))
    np.save(bp, b.astype(np.float32))