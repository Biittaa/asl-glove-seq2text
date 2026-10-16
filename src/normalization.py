from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def compute_feature_stats(dataset_root, train_split, feature_columns) -> dict:
    import csv
    manifest_path = Path(dataset_root) / "manifests" / f"{train_split}.jsonl"
    sums = np.zeros(len(feature_columns))
    sq_sums = np.zeros(len(feature_columns))
    count = 0
    with open(manifest_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    for record in records:
        path = Path(dataset_root) / record["sequence_path"]
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals = np.array([float(row[c]) for c in feature_columns])
                sums += vals
                sq_sums += vals ** 2
                count += 1
    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_stats(stats: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32)


def fuzzy_encode(x: np.ndarray, n_levels: int = 7, clip: float = 3.0) -> np.ndarray:
    """Fuzzy encoding: snap a normalized signal onto a few discrete levels.

    Idea borrowed from Wei et al. (InfoMat 2024): after normalization the
    signal is converted to a small number of equally spaced magnitude levels,
    so the waveform SHAPE is kept while small amplitude differences between
    signers / repetitions are removed.

    x       : already normalized array (e.g. per-recording z-score), any shape.
    n_levels: number of discrete levels (the paper uses 6 for strain sensors).
              An odd value keeps an exact 0 level.
    clip    : values outside [-clip, clip] are saturated before quantization.

    Returns an array in [-1, 1] that takes only `n_levels` distinct values.
    """
    if n_levels < 2:
        return x.astype(np.float32)
    x = np.clip(x, -clip, clip) / clip               # -> [-1, 1]
    u = (x + 1.0) / 2.0                              # -> [0, 1]
    q = np.round(u * (n_levels - 1)) / (n_levels - 1)  # snap to equally spaced levels
    return (q * 2.0 - 1.0).astype(np.float32)        # -> [-1, 1]


def normalize_recording(
    arr: np.ndarray,
    mode: str,
    mean=None,
    std=None,
    eps: float = 1e-3,
    fuzzy_levels: int = 0,
    fuzzy_clip: float = 3.0,
) -> np.ndarray:
    """Normalize one recording, then optionally apply fuzzy encoding.

    fuzzy_levels <= 1 disables fuzzy encoding (old behaviour).
    """
    if mode == "recording":
        m = arr.mean(axis=0, keepdims=True)
        s = arr.std(axis=0, keepdims=True)
        out = (arr - m) / np.maximum(s, eps)
    elif mode == "recording_center":
        out = (arr - arr.mean(axis=0, keepdims=True)) / std
    else:
        out = (arr - mean) / std

    if fuzzy_levels and fuzzy_levels > 1:
        out = fuzzy_encode(out, fuzzy_levels, fuzzy_clip)
    return out