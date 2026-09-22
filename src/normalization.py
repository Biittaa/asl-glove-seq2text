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