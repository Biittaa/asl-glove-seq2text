"""Transformer encoder-decoder for sensor-sequence -> dual text prediction.

This follows the reference architecture (encoder: PE -> Transformer Encoder ->
Linear+Softmax -> CTC; decoder: word embedding + PE -> Transformer Decoder ->
Linear+Softmax), with a lightweight CNN front-end (``ConvSubsampling`` below)
placed before the Transformer encoder instead of the paper's full CNN+GRU
"sign embedding" (SE) block. Its only job is to shrink the frame axis before
self-attention: Transformer self-attention costs O(T^2 * d_model), so raw
per-frame sequences of several hundred frames dominate both training time and
memory. Two stride-2 1D-conv layers cut T by 4x (``conv_num_layers`` /
``conv_stride`` in the config control this), so attention cost drops by
roughly 16x while the GRU (which existed in the paper mainly to build
frame-local temporal context, something the conv stack already does more
cheaply) is intentionally left out. Set ``use_conv_subsampling=False`` to
fall back to the previous behaviour: a single linear projection
(``FeatureEmbedding``) with no downsampling, i.e. f_k = Linear(raw_frame_k).

Because subsampling shortens the frame axis, the encoder output length K is
now < the input frame count T (K ~= T / conv_stride**conv_num_layers rather
than K == T). The downsampled per-sample lengths and padding mask are
computed alongside the conv stack (see ``ConvSubsampling.forward``) and
returned from ``TransformerEncoderCTC``/``SignLanguageTransformer`` so the
CTC loss and CTC greedy decode -- which need to know the true, unpadded
length of each sample in the *encoder's* output space, not the input frame
count -- use the right lengths. Encoder outputs Z = {z_1..z_K} still double
as both the CTC input sequence and the memory the decoder cross-attends to,
matching the arrow from the encoder stack to the decoder stack in the
reference figure; the decoder itself is unaffected by subsampling.
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


class ConvSubsampling(nn.Module):
    """Stride-2 1D-conv stack that downsamples the frame axis before attention.

    Stacking ``num_layers`` conv layers of stride ``stride`` shrinks the
    sequence length by ``stride ** num_layers`` (default 2**2 = 4x), which
    shrinks Transformer self-attention cost (O(T^2)) by the square of that,
    i.e. ~16x by default. Kernel size > 1 also lets each output step see a
    small window of neighbouring raw frames, similar in spirit to (but much
    cheaper than) the paper's CNN+GRU sign-embedding front-end.

    Alongside the downsampled features, this module also downsamples the
    per-sample true lengths using the standard conv output-length formula,
    so callers can build an exact padding mask for the new, shorter sequence
    (needed for the Transformer encoder's key_padding_mask, for CTC's
    input_lengths, and for CTC greedy decoding).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_size: int = 5,
        stride: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = d_model
        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _subsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths
        for _ in range(self.num_layers):
            out = torch.div(
                out + 2 * self.padding - self.kernel_size, self.stride, rounding_mode="floor"
            ) + 1
        return out.clamp(min=1)

    def forward(
        self, features: torch.Tensor, frame_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: (B, T, input_dim); frame_lengths: (B,) true (unpadded) frame counts.
        x = features.transpose(1, 2)          # (B, input_dim, T)
        x = self.conv(x)                       # (B, d_model, T')
        x = x.transpose(1, 2)                  # (B, T', d_model)
        x = self.dropout(self.norm(x))

        out_lengths = self._subsampled_lengths(frame_lengths)          # (B,)
        max_len = x.size(1)
        key_padding_mask = (
            torch.arange(max_len, device=x.device)[None, :] >= out_lengths[:, None]
        )  # (B, T') bool, True = pad
        return x, out_lengths, key_padding_mask


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
        use_conv_subsampling: bool = True,
        conv_kernel_size: int = 5,
        conv_stride: int = 2,
        conv_num_layers: int = 2,
        two_hand: bool = False,
        hand_feature_dim: int = 19,
        hand_d_model: int = 128,
        fusion: str = "concat",
        fusion_nhead: int = 4,
        hand_encoder_layers: int = 0,
    ) -> None:
        super().__init__()
        self.use_conv_subsampling = use_conv_subsampling or two_hand
        if two_hand:
            self.subsample = HandFusionFrontEnd(
                hand_feature_dim=hand_feature_dim,
                hand_d_model=hand_d_model,
                d_model=d_model,
                fusion=fusion,
                nhead=fusion_nhead,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                num_layers=conv_num_layers,
                dropout=dropout,
                hand_layers=hand_encoder_layers,
                max_len=max_len,
            )
        elif use_conv_subsampling:
            self.subsample = ConvSubsampling(
                input_feature_dim,
                d_model,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                num_layers=conv_num_layers,
                dropout=dropout,
            )
        else:
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
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: (B, T, input_feature_dim); src_key_padding_mask: (B, T) bool, True = pad
        # frame_lengths: (B,) true (unpadded) input frame counts.
        if self.use_conv_subsampling:
            f, out_lengths, key_padding_mask = self.subsample(features, frame_lengths)  # (B, T', d_model)
        else:
            f = self.feature_embed(features)            # f_k^E  (linear, replaces SE)
            out_lengths = frame_lengths
            key_padding_mask = src_key_padding_mask
        f = self.pos_encoding(f)                     # f_k = f_k^E + PE(k)
        z = self.encoder(f, src_key_padding_mask=key_padding_mask)  # Z = {z_k}, k in [1, T']
        ctc_logits = self.ctc_head(z)                 # p(w'_k | F^E) pre-softmax
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)
        return z, ctc_log_probs, key_padding_mask, out_lengths


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
        use_conv_subsampling: bool = True,
        conv_kernel_size: int = 5,
        conv_stride: int = 2,
        conv_num_layers: int = 2,
        two_hand: bool = False,
        hand_feature_dim: int = 19,
        hand_d_model: int = 128,
        fusion: str = "concat",
        fusion_nhead: int = 4,
        hand_encoder_layers: int = 0,
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
            use_conv_subsampling=use_conv_subsampling,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_num_layers=conv_num_layers,
            two_hand=two_hand,
            hand_feature_dim=hand_feature_dim,
            hand_d_model=hand_d_model,
            fusion=fusion,
            fusion_nhead=fusion_nhead,
            hand_encoder_layers=hand_encoder_layers,
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
        frame_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory, ctc_log_probs, encoder_key_padding_mask, encoder_frame_lengths = self.encoder(
            features, src_key_padding_mask, frame_lengths
        )

        # Teacher forcing: feed all tokens except the last (<eos>), predict
        # all tokens except the first (<bos>) -> standard shifted setup.
        tgt_in = word_ids[:, :-1]
        tgt_key_padding_mask = tgt_in.eq(self.word_pad_id)
        decoder_logits = self.decoder(
            tgt_in=tgt_in,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_key_padding_mask,
        )
        return {
            "ctc_log_probs": ctc_log_probs,       # (B, T', gloss_vocab_size) -- T' after subsampling
            "decoder_logits": decoder_logits,     # (B, L-1, word_vocab_size)
            "memory": memory,
            # Needed downstream (CTC loss / CTC greedy decode) because T' != the
            # original input frame count whenever conv subsampling is enabled.
            "encoder_key_padding_mask": encoder_key_padding_mask,   # (B, T') bool, True = pad
            "encoder_frame_lengths": encoder_frame_lengths,         # (B,) true lengths in encoder's T' space
        }

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        frame_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive greedy decoding for inference on unseen sequences.

        ``frame_lengths`` is optional: when omitted, it is derived from
        ``src_key_padding_mask`` (true frame count = number of non-pad
        positions), which is convenient for single-sample inference where
        there is no padding at all (see ``infer.py``).
        """
        self.eval()
        device = features.device
        batch_size = features.size(0)
        if frame_lengths is None:
            frame_lengths = (~src_key_padding_mask).sum(dim=1)
        memory, _, encoder_key_padding_mask, _ = self.encoder(features, src_key_padding_mask, frame_lengths)

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_key_padding_mask = generated.eq(self.word_pad_id)
            logits = self.decoder(
                tgt_in=generated,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=encoder_key_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            next_token = torch.where(finished, torch.full_like(next_token, self.word_pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if bool(finished.all()):
                break

        return generated  # (B, <=max_len), includes <bos> and (if reached) <eos>


class HandFusionFrontEnd(nn.Module):

    def __init__(self, hand_feature_dim, hand_d_model, d_model, fusion,
                 nhead, kernel_size, stride, num_layers, dropout, hand_layers=0, max_len=1024):
        super().__init__()
        if fusion not in ("concat", "cross_attention"):
            raise ValueError(f"Unknown fusion: {fusion}")
        self.hand_dim = hand_feature_dim
        self.fusion = fusion

        def make():
            return ConvSubsampling(hand_feature_dim, hand_d_model,
                                   kernel_size, stride, num_layers, dropout)
        self.right, self.left = make(), make()

        if fusion == "cross_attention":
            self.attn_r = nn.MultiheadAttention(hand_d_model, nhead, dropout=dropout, batch_first=True)
            self.attn_l = nn.MultiheadAttention(hand_d_model, nhead, dropout=dropout, batch_first=True)

        
        self.hand_layers = hand_layers
        if hand_layers > 0:
            def make_enc():
                layer = nn.TransformerEncoderLayer(
                    d_model=hand_d_model, nhead=nhead, dim_feedforward=hand_d_model * 2,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
                return nn.TransformerEncoder(layer, num_layers=hand_layers, enable_nested_tensor=False)
            self.enc_r, self.enc_l = make_enc(), make_enc()
            self.pe = PositionalEncoding(hand_d_model, dropout, max_len)
        
        self.out_proj = nn.Linear(2 * hand_d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, features, frame_lengths):
        h = self.hand_dim
        r, out_lengths, mask = self.right(features[..., :h], frame_lengths)   # (B, T', hd)
        l, _, _ = self.left(features[..., h:], frame_lengths)                  # (B, T', hd)
        if self.hand_layers > 0:
            r = self.enc_r(self.pe(r), src_key_padding_mask=mask)
            l = self.enc_l(self.pe(l), src_key_padding_mask=mask)
        if self.fusion == "cross_attention":
            r2, _ = self.attn_r(r, l, l, key_padding_mask=mask, need_weights=False)
            l2, _ = self.attn_l(l, r, r, key_padding_mask=mask, need_weights=False)
            r, l = r + r2, l + l2          # residual
        x = self.out_norm(self.out_proj(torch.cat([r, l], dim=-1)))           # (B, T', d_model)
        return x, out_lengths, mask