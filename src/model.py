"""Transformer encoder-decoder for sensor-sequence -> dual text prediction.

This follows the reference architecture (encoder: PE -> Transformer Encoder ->
Linear+Softmax -> CTC; decoder: word embedding + PE -> Transformer Decoder ->
Linear+Softmax), with ONE deliberate change requested by the user: the
CNN+GRU "sign embedding" (SE) block that precedes the encoder in the paper is
REMOVED. The model therefore starts directly at the Transformer encoder; raw
per-frame sensor features are mapped to ``d_model`` with a single linear
projection (``FeatureEmbedding`` below) before positional encoding is added,
i.e. f_k = Linear(raw_frame_k) instead of f_k^E = SE(raw_frame_k).

Because there is no CNN downsampling, the encoder output length K equals the
input frame count T, so encoder outputs Z = {z_1..z_K} double as both the CTC
input sequence and the memory the decoder cross-attends to (matching the
arrow from the encoder stack to the decoder stack in the reference figure).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class FeatureEmbedding(nn.Module):
    """Linear projection of raw per-frame sensor features to ``d_model``.

    This stands in for the paper's CNN+GRU sign-embedding (SE) block, which
    was intentionally removed; only a linear map + LayerNorm is used here.
    """

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class TransformerEncoderCTC(nn.Module):
    """Frame-level Transformer encoder + CTC classification head.

    Produces both:
      - ``memory`` Z = {z_k}: the contextual encoder outputs, used as the
        decoder's cross-attention memory.
      - ``ctc_log_probs``: per-frame log-probabilities over the gloss
        vocabulary (including the CTC blank at index 0), used for L_CTC
        against the gesture-order (``label_no_order``) target.
    """

    def __init__(
        self,
        input_feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        gloss_vocab_size: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.feature_embed = FeatureEmbedding(input_feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ctc_head = nn.Linear(d_model, gloss_vocab_size)

    def forward(
        self, features: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (B, T, input_feature_dim); src_key_padding_mask: (B, T) bool, True = pad
        f = self.feature_embed(features)            # f_k^E  (linear, replaces SE)
        f = self.pos_encoding(f)                     # f_k = f_k^E + PE(k)
        z = self.encoder(f, src_key_padding_mask=src_key_padding_mask)  # Z = {z_k}
        ctc_logits = self.ctc_head(z)                 # p(w'_k | F^E) pre-softmax
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
        return z, ctc_log_probs


class TransformerDecoderCE(nn.Module):
    """Autoregressive Transformer decoder + cross-entropy head.

    Predicts the spoken-order (``label_order``) sequence token-by-token from
    the encoder memory Z, using teacher forcing during training.
    """

    def __init__(
        self,
        word_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_id: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.word_embed = nn.Embedding(word_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, word_vocab_size)

    @staticmethod
    def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        # (size, size) additive mask: 0 where attention is allowed, -inf where forbidden.
        mask = torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(
        self,
        tgt_in: torch.Tensor,                 # (B, L) token ids fed to the decoder
        memory: torch.Tensor,                 # (B, T, d_model) encoder output Z
        tgt_key_padding_mask: torch.Tensor,   # (B, L) bool, True = pad
        memory_key_padding_mask: torch.Tensor,  # (B, T) bool, True = pad
    ) -> torch.Tensor:
        w = self.word_embed(tgt_in) * math.sqrt(self.d_model)  # w_m = WE(w_m)
        w = self.pos_encoding(w)                                 # + PE(m)
        causal_mask = self.generate_causal_mask(tgt_in.size(1), tgt_in.device)
        h = self.decoder(
            tgt=w,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.out_proj(h)  # y_m pre-softmax; CrossEntropyLoss applies log-softmax internally
        return logits


class SignLanguageTransformer(nn.Module):
    """Full model: Transformer encoder (+CTC head) feeding a Transformer decoder (+CE head)."""

    def __init__(
        self,
        input_feature_dim: int,
        gloss_vocab_size: int,
        word_vocab_size: int,
        word_pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_encoder_len: int = 1024,
        max_decoder_len: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoderCTC(
            input_feature_dim=input_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            gloss_vocab_size=gloss_vocab_size,
            max_len=max_encoder_len,
        )
        self.decoder = TransformerDecoderCE(
            word_vocab_size=word_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_id=word_pad_id,
            max_len=max_decoder_len,
        )
        self.word_pad_id = word_pad_id

    def forward(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory, ctc_log_probs = self.encoder(features, src_key_padding_mask)

        # Teacher forcing: feed all tokens except the last (<eos>), predict
        # all tokens except the first (<bos>) -> standard shifted setup.
        tgt_in = word_ids[:, :-1]
        tgt_key_padding_mask = tgt_in.eq(self.word_pad_id)
        decoder_logits = self.decoder(
            tgt_in=tgt_in,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return {
            "ctc_log_probs": ctc_log_probs,       # (B, T, gloss_vocab_size)
            "decoder_logits": decoder_logits,     # (B, L-1, word_vocab_size)
            "memory": memory,
        }

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """Autoregressive greedy decoding for inference on unseen sequences."""
        self.eval()
        device = features.device
        batch_size = features.size(0)
        memory, _ = self.encoder(features, src_key_padding_mask)

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_key_padding_mask = generated.eq(self.word_pad_id)
            logits = self.decoder(
                tgt_in=generated,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            next_token = torch.where(finished, torch.full_like(next_token, self.word_pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if bool(finished.all()):
                break

        return generated  # (B, <=max_len), includes <bos> and (if reached) <eos>
