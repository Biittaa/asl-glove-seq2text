"""Dataset for the ASL Perception Neuron sensor-to-text package.
...
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .normalization import compute_feature_stats, load_stats, save_stats
from .vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocabulary


class ASLManifestDataset(torch.utils.data.Dataset):
    """Reads one manifest split (train/validation/test) and its CSV sequences."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        gloss_vocab: Vocabulary,
        word_vocab: Vocabulary,
        manifest_subdir: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.gloss_vocab = gloss_vocab
        self.word_vocab = word_vocab

        manifest_path = self.dataset_root / "manifests" / manifest_subdir/ f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        metadata_path = self.dataset_root / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.feature_columns: list[str] = list(self.metadata["feature_columns"])

        # --- feature normalization stats (built once from train, cached) ---
        stats_path = self.dataset_root / "artifacts" / manifest_subdir / "feature_stats.json"
        if stats_path.is_file():
            self.feat_mean, self.feat_std = load_stats(stats_path)
        else:
            stats = compute_feature_stats(self.dataset_root, f"{manifest_subdir}/train", self.feature_columns)
            save_stats(stats, stats_path)
            self.feat_mean = np.array(stats["mean"], dtype=np.float32)
            self.feat_std = np.array(stats["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def _load_sequence(self, relative_path: str) -> np.ndarray:
        sequence_path = self.dataset_root / relative_path
        expected_header = ["timestamp", *self.feature_columns]
        rows: list[list[float]] = []
        with sequence_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != expected_header:
                raise ValueError(f"Unexpected columns in {sequence_path}")
            for row in reader:
                rows.append([float(row[name]) for name in self.feature_columns])
        arr = np.asarray(rows, dtype=np.float32)
        arr = (arr - self.feat_mean) / self.feat_std
        return arr

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        features = self._load_sequence(record["sequence_path"])  # (T, 38)

        gloss_ids = self.gloss_vocab.encode(record["label_no_order"].split())

        word_tokens = [BOS_TOKEN, *record["label_order"].split(), EOS_TOKEN]
        word_ids = self.word_vocab.encode(word_tokens)

        return {
            "id": record["id"],
            "features": torch.from_numpy(features),          # (T, 38) float32
            "num_frames": features.shape[0],
            "gloss_ids": torch.tensor(gloss_ids, dtype=torch.long),   # (Lg,)
            "word_ids": torch.tensor(word_ids, dtype=torch.long),     # (Lw,) incl. <bos>/<eos>
        }


def collate_asl_batch(batch: list[dict[str, Any]], word_pad_id: int) -> dict[str, Any]:
    """Pads a batch and builds the masks/lengths needed by the model and losses."""
    ids = [item["id"] for item in batch]

    frame_lengths = torch.tensor([item["num_frames"] for item in batch], dtype=torch.long)
    max_frames = int(frame_lengths.max().item())

    features = pad_sequence([item["features"] for item in batch], batch_first=True)  # (B, Tmax, 38)
    src_key_padding_mask = torch.arange(max_frames)[None, :] >= frame_lengths[:, None]

    gloss_lengths = torch.tensor([item["gloss_ids"].shape[0] for item in batch], dtype=torch.long)
    gloss_targets_concat = torch.cat([item["gloss_ids"] for item in batch])

    word_ids_padded = pad_sequence(
        [item["word_ids"] for item in batch], batch_first=True, padding_value=word_pad_id
    )  # (B, Lmax)

    return {
        "ids": ids,
        "features": features,
        "frame_lengths": frame_lengths,
        "src_key_padding_mask": src_key_padding_mask,
        "gloss_targets_concat": gloss_targets_concat,
        "gloss_lengths": gloss_lengths,
        "word_ids": word_ids_padded,
    }