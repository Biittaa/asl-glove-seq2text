"""One epoch of training, and a validation/test pass with WER computation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .losses import JointCTCCrossEntropyLoss
from .metrics import ctc_greedy_decode_ids, sentence_accuracy, word_error_rate, word_level_accuracy
from .model import SignLanguageTransformer
from .vocab import Vocabulary


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ("features", "frame_lengths", "src_key_padding_mask",
                "gloss_targets_concat", "gloss_lengths", "word_ids"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def train_one_epoch(
    model: SignLanguageTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: JointCTCCrossEntropyLoss,
    device: torch.device,
    grad_clip_norm: float,
    scaler: torch.cuda.amp.GradScaler | None,
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_ctc": 0.0, "loss_ce": 0.0}
    num_batches = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out = model(
                batch["features"], batch["src_key_padding_mask"], batch["word_ids"], batch["frame_lengths"]
            )
            losses = criterion(
                ctc_log_probs=out["ctc_log_probs"],
                # Use the encoder's own (possibly subsampled) per-sample
                # lengths here, NOT batch["frame_lengths"]: ctc_log_probs has
                # T' frames (T' == T only when conv subsampling is off), and
                # nn.CTCLoss requires input_lengths to match that axis.
                frame_lengths=out["encoder_frame_lengths"],
                gloss_targets_concat=batch["gloss_targets_concat"],
                gloss_lengths=batch["gloss_lengths"],
                decoder_logits=out["decoder_logits"],
                word_targets=batch["word_ids"],
            )

        loss = losses["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        for key in running:
            running[key] += float(losses[key].item()) if key != "loss" else float(loss.item())
        num_batches += 1

        if step % log_every == 0:
            print(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss={running['loss'] / num_batches:.4f} "
                f"ctc={running['loss_ctc'] / num_batches:.4f} "
                f"ce={running['loss_ce'] / num_batches:.4f}"
            )

    return {key: value / max(num_batches, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate(
    model: SignLanguageTransformer,
    loader: DataLoader,
    criterion: JointCTCCrossEntropyLoss,
    device: torch.device,
    gloss_vocab: Vocabulary,
    word_vocab: Vocabulary,
    max_decode_len: int,
    split: str = "val",
) -> dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_ctc": 0.0, "loss_ce": 0.0}
    num_batches = 0

    gloss_refs, gloss_hyps = [], []
    word_refs, word_hyps = [], []

    bos_id = word_vocab.stoi["<bos>"]
    eos_id = word_vocab.stoi["<eos>"]
    blank_id = gloss_vocab.stoi["<blank>"]

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            batch["features"], batch["src_key_padding_mask"], batch["word_ids"], batch["frame_lengths"]
        )
        losses = criterion(
            ctc_log_probs=out["ctc_log_probs"],
            # Same reasoning as in train_one_epoch: match ctc_log_probs' T'.
            frame_lengths=out["encoder_frame_lengths"],
            gloss_targets_concat=batch["gloss_targets_concat"],
            gloss_lengths=batch["gloss_lengths"],
            decoder_logits=out["decoder_logits"],
            word_targets=batch["word_ids"],
        )
        for key in running:
            running[key] += float(losses[key].item()) if key != "loss" else float(losses["loss"].item())
        num_batches += 1

        # --- CTC (gloss) greedy decode for reference/monitoring ---
        gloss_hyp_ids = ctc_greedy_decode_ids(out["ctc_log_probs"], out["encoder_frame_lengths"], blank_id)
        gloss_hyps.extend([gloss_vocab.decode(ids) for ids in gloss_hyp_ids])
        offset = 0
        for length in batch["gloss_lengths"].tolist():
            ref_ids = batch["gloss_targets_concat"][offset: offset + length].tolist()
            gloss_refs.append(gloss_vocab.decode(ref_ids))
            offset += length

        # --- Decoder (spoken order) greedy decode ---
        generated = model.greedy_decode(
            batch["features"], batch["src_key_padding_mask"], bos_id, eos_id, max_decode_len,
            frame_lengths=batch["frame_lengths"],
        )
        for row in generated.tolist():
            word_hyps.append(word_vocab.decode(row[1:], stop_at_eos=True))  # drop <bos>
        for row in batch["word_ids"].tolist():
            word_refs.append(word_vocab.decode(row[1:], stop_at_eos=True))  # drop <bos>, stop at <eos>
    metrics = {key: value / max(num_batches, 1) for key, value in running.items()}

    # --- gloss (CTC) metrics: word-level و sentence-level ---
    metrics[f"{split}_gloss_wer"] = word_error_rate(gloss_refs, gloss_hyps)
    metrics[f"{split}_gloss_word_acc"] = word_level_accuracy(gloss_refs, gloss_hyps)
    metrics[f"{split}_gloss_sentence_acc"] = sentence_accuracy(gloss_refs, gloss_hyps)

    # --- decoder (spoken order) metrics: word-level و sentence-level ---
    metrics[f"{split}_wer"] = word_error_rate(word_refs, word_hyps)
    metrics[f"{split}_word_acc"] = word_level_accuracy(word_refs, word_hyps)
    metrics[f"{split}_sentence_acc"] = sentence_accuracy(word_refs, word_hyps)

    return metrics
    # metrics = {key: value / max(num_batches, 1) for key, value in running.items()}
    # metrics["val_gloss_wer"] = word_error_rate(gloss_refs, gloss_hyps)
    # metrics["val_wer"] = word_error_rate(word_refs, word_hyps)
    # return metrics
