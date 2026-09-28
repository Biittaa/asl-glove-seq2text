#!/usr/bin/env python3
"""Train the sensor-to-text Transformer (encoder+CTC / decoder+CE).

Usage (from the repo root, e.g. inside the cloned GitHub repo on Colab):

    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --dataset_root /content/drive/MyDrive/.../asl_training_dataset

Any dotted CLI override (e.g. --train.batch_size 32) takes precedence over
the YAML config, so you don't have to edit the file for quick experiments.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import ASLManifestDataset, collate_asl_batch
from src.engine import evaluate, train_one_epoch
from src.losses import JointCTCCrossEntropyLoss
from src.model import SignLanguageTransformer
from src.utils import WarmupInverseSqrtLR, load_checkpoint, resolve_device, save_checkpoint, set_seed
from src.vocab import build_or_load_vocabs


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override data.dataset_root")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--split_mode", type=str, default=None, choices=["subject", "random", "compositional"], help="Override data.split.mode")
    return parser.parse_args()


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dataset_root:
        cfg["data"]["dataset_root"] = args.dataset_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.split_mode:
        cfg["data"]["split"]["mode"] = args.split_mode
    mode = cfg["data"]["split"]["mode"]
    cfg["vocab"]["out_dir"] = f"{cfg['vocab']['out_dir']}/{mode}"
    cfg["train"]["ckpt_dir"] = f"{cfg['train']['ckpt_dir']}/{mode}"
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    set_seed(cfg["train"]["seed"])
    device = resolve_device(cfg["runtime"]["device"])
    print(f"[info] using device: {device}")

    dataset_root = cfg["data"]["dataset_root"]
    mode = cfg["data"]["split"]["mode"]
    gloss_vocab, word_vocab = build_or_load_vocabs(
        dataset_root=dataset_root,
        train_split=f"{mode}/{cfg['data']['train_split']}",
        out_dir=cfg["vocab"]["out_dir"],
        gloss_vocab_file=cfg["vocab"]["gloss_vocab_file"],
        word_vocab_file=cfg["vocab"]["word_vocab_file"],
    )
    print(f"[info] gloss (CTC) vocab size: {len(gloss_vocab)}")
    print(f"[info] word (decoder) vocab size: {len(word_vocab)}")

    word_pad_id = word_vocab.stoi["<pad>"]
    collate_fn = partial(collate_asl_batch, word_pad_id=word_pad_id)

    train_set = ASLManifestDataset(dataset_root, cfg["data"]["train_split"], gloss_vocab, word_vocab, mode)
    val_set = ASLManifestDataset(dataset_root, cfg["data"]["val_split"], gloss_vocab, word_vocab, mode)
    test_set = ASLManifestDataset(dataset_root, cfg["data"]["test_split"], gloss_vocab, word_vocab, mode)
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=collate_fn,
    )

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

    criterion = JointCTCCrossEntropyLoss(
        blank_id=gloss_vocab.stoi["<blank>"],
        pad_id=word_pad_id,
        lambda_ctc=cfg["loss"]["lambda_ctc"],
        lambda_ce=cfg["loss"]["lambda_ce"],
        label_smoothing=cfg["loss"]["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = WarmupInverseSqrtLR(optimizer, cfg["model"]["d_model"], cfg["train"]["warmup_steps"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["amp"] and device.type == "cuda") else None

    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric_value = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            cfg["train"]["grad_clip_norm"], scaler, cfg["train"]["log_every"], epoch,
        )
        print(f"[epoch {epoch}] train: {train_metrics}")

        if epoch % cfg["train"]["eval_every_epochs"] == 0:
            val_metrics = evaluate(
                model, val_loader, criterion, device, gloss_vocab, word_vocab,
                cfg["model"]["max_decoder_len"], split="val",
            )
            print(f"[epoch {epoch}] val: {val_metrics}")
            
            test_metrics = evaluate(
                model, test_loader, criterion, device, gloss_vocab, word_vocab,
                cfg["model"]["max_decoder_len"], split="test",
            )
            print(f"[epoch {epoch}] test: {test_metrics}")

            metric_value = val_metrics[cfg["train"]["best_metric"]]
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch,
                             {"val_metrics": val_metrics, "test_metrics": test_metrics})
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                epochs_without_improvement = 0
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch,
                                 {"val_metrics": val_metrics, "test_metrics": test_metrics})
                print(f"[epoch {epoch}] new best {cfg['train']['best_metric']}={metric_value:.4f}, checkpoint saved.")
            else:
                epochs_without_improvement += cfg["train"]["eval_every_epochs"]

            patience = cfg["train"]["early_stop_patience"]
            if patience and epochs_without_improvement >= patience:
                print(f"[info] no improvement for {epochs_without_improvement} epochs, stopping early.")
                break

    print("[info] training complete.")
    
    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.is_file():
        load_checkpoint(best_ckpt_path, model)
        print(f"[info] loaded best checkpoint from {best_ckpt_path} for final test evaluation.")
    else:
        print("[warn] no best.pt found, evaluating the final in-memory model weights instead.")

    final_test_metrics = evaluate(
        model, test_loader, criterion, device, gloss_vocab, word_vocab,
        cfg["model"]["max_decoder_len"], split="test",
    )
    print("[info] ===== FINAL TEST RESULTS (best checkpoint) =====")
    for key, value in final_test_metrics.items():
        print(f"[final test] {key}: {value:.4f}")


if __name__ == "__main__":
    main()
