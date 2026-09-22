"""Joint loss: L = lambda_ctc * L_CTC + lambda_ce * L_CE.

- L_CTC is computed between the encoder's per-frame log-probabilities and the
  GESTURE-order label (``label_no_order``): this is the label that has no
  fixed word order, matching the paper's use of CTC to align an unsegmented
  frame sequence to an (order-free / monotonic) gloss annotation.
- L_CE is computed between the decoder's per-step logits and the
  SPOKEN-order label (``label_order``), i.e. the label that IS in the
  correct target sentence order.
"""

from __future__ import annotations

import torch
from torch import nn


class JointCTCCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        blank_id: int,
        pad_id: int,
        lambda_ctc: float,
        lambda_ce: float,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.lambda_ce = lambda_ce
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    def forward(
        self,
        ctc_log_probs: torch.Tensor,     # (B, T, V_gloss)
        frame_lengths: torch.Tensor,     # (B,) true (unpadded) frame counts
        gloss_targets_concat: torch.Tensor,  # (sum(Lg),) concatenated gloss ids
        gloss_lengths: torch.Tensor,     # (B,)
        decoder_logits: torch.Tensor,    # (B, L-1, V_word)
        word_targets: torch.Tensor,      # (B, L) includes <bos>/<eos>, target = word_targets[:, 1:]
    ) -> dict[str, torch.Tensor]:
        # nn.CTCLoss expects (T, B, V) log-probabilities.
        ctc_input = ctc_log_probs.transpose(0, 1)  # (T, B, V_gloss)
        l_ctc = self.ctc_loss(ctc_input, gloss_targets_concat, frame_lengths, gloss_lengths)

        ce_target = word_targets[:, 1:]  # drop <bos>; predict each next token including <eos>
        l_ce = self.ce_loss(
            decoder_logits.reshape(-1, decoder_logits.size(-1)),
            ce_target.reshape(-1),
        )

        total = self.lambda_ctc * l_ctc + self.lambda_ce * l_ce
        return {"loss": total, "loss_ctc": l_ctc.detach(), "loss_ce": l_ce.detach()}
