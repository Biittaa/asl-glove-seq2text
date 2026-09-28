#!/usr/bin/env python3
"""Run inference on a single sequence CSV (e.g. a brand-new, unseen recording).

Usage:
    python infer.py --config configs/config.yaml --checkpoint artifacts/checkpoints/best.pt \
        --csv_path /path/to/some_new_recording.csv
"""

from __future__ import annotations

import argparse
import csv
import json

import torch
import yaml

from src.model import SignLanguageTransformer
from src.utils import load_checkpoint, resolve_device
from src.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    return parser.parse_args()


def load_sequence(csv_path: str, feature_columns: list[str]) -> torch.Tensor:
    expected_header = ["timestamp", *feature_columns]
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_header:
            raise ValueError(f"Unexpected columns in {csv_path}")
        for row in reader:
            rows.append([float(row[name]) for name in feature_columns])
    return torch.tensor(rows, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg["runtime"]["device"])
    mode = cfg["data"]["split"]["mode"]
    
    gloss_vocab = Vocabulary.from_json(f"{cfg['vocab']['out_dir']}/{mode}/{cfg['vocab']['gloss_vocab_file']}")
    word_vocab = Vocabulary.from_json(f"{cfg['vocab']['out_dir']}/{mode}/{cfg['vocab']['word_vocab_file']}")
    
    word_pad_id = word_vocab.stoi["<pad>"]

    with open(f"{cfg['data']['dataset_root']}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    feature_columns = list(metadata["feature_columns"])

    model = SignLanguageTransformer(
        input_feature_dim=cfg["model"]["input_feature_dim"],
        gloss_vocab_size=len(gloss_vocab),
        word_vocab_size=len(word_vocab),
        word_pad_id=word_pad_id,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        encoder_layers=cfg["model"]["encoder_layers"],
        decoder_layers=cfg["model"]["decoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        max_encoder_len=cfg["model"]["max_encoder_len"],
        max_decoder_len=cfg["model"]["max_decoder_len"],
        use_conv_subsampling=cfg["model"].get("use_conv_subsampling", True),
        conv_kernel_size=cfg["model"].get("conv_kernel_size", 5),
        conv_stride=cfg["model"].get("conv_stride", 2),
        conv_num_layers=cfg["model"].get("conv_num_layers", 2),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device.type)
    model.eval()

    features = load_sequence(args.csv_path, feature_columns).unsqueeze(0).to(device)  # (1, T, 38)
    src_key_padding_mask = torch.zeros(1, features.size(1), dtype=torch.bool, device=device)
    frame_lengths = torch.tensor([features.size(1)], dtype=torch.long, device=device)

    bos_id, eos_id = word_vocab.stoi["<bos>"], word_vocab.stoi["<eos>"]
    generated = model.greedy_decode(
        features, src_key_padding_mask, bos_id, eos_id, cfg["model"]["max_decoder_len"],
        frame_lengths=frame_lengths,
    )
    words = word_vocab.decode(generated[0, 1:].tolist(), stop_at_eos=True)
    print("Predicted spoken-order sentence:", " ".join(words))


if __name__ == "__main__":
    main()
