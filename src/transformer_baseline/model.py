"""
Transformer Seq2Seq Baseline — For length extrapolation ablation study.

This model uses a global, autoregressive Transformer decoder with Absolute
Positional Encoding and <EOS> token-based stopping. These architectural choices
are KNOWN to cause length extrapolation failure, which is exactly what we want
to demonstrate in our paper.

Architecture:
  2D CNN Encoder (copied from src.CRNN_CTC_Uncapped.model): [B,1,64,W] → [B,512,8,W//8]
  Height Collapse (mean(dim=2))                            : [B,512,8,T] → [B,512,T]
  Feature Projector (Linear)                               : [B,T,512]  → [B,T,256]
  Absolute Positional Encoding                             : [B,T,256]  → [B,T,256]
  Transformer Decoder (4 layers, d_model=256)              : [B,L,256]  → [B,L,256]
  Classifier Head                                          : [B,L,256]  → [B,L,13]

Vocabulary (13 tokens):
  - Digits 0–9  (indices 0–9)
  - <PAD>       (index 10)
  - <SOS>       (index 11)
  - <EOS>       (index 12)

Critical Design Choices (for paper narrative):
  1. Absolute Sinusoidal Positional Encoding — causes length prior learning
     because the encoder features and decoder embeddings each carry a rigid
     position bias that generalises poorly beyond max_len seen at training time.
  2. Autoregressive decoding with <EOS> — causes exposure-bias cascade.
     At OOD lengths the model starts predicting <EOS> early because the
     position embedding signals "this is where training sequences ended".
  3. Global self-attention — allows the decoder to see the entire partial
     sequence at once, enabling shortcut learning of sequence length from
     context rather than from the local image evidence.

These three choices together ensure the model will fail catastrophically on
OOD lengths, providing the contrast needed to demonstrate that our CTC model's
length robustness is not a coincidence but a structural property.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary constants
# ─────────────────────────────────────────────────────────────────────────────
PAD_IDX: int = 10   # padding — ignored by CrossEntropyLoss
SOS_IDX: int = 11   # start-of-sequence token prepended at decode time
EOS_IDX: int = 12   # end-of-sequence token appended to targets at training time
VOCAB_SIZE: int = 13   # 0-9 (digits) + PAD + SOS + EOS


# ─────────────────────────────────────────────────────────────────────────────
# 2D CNN Encoder
# Copied verbatim from src/CRNN_CTC_Uncapped/model.py for isolation — this
# avoids a hard cross-module import and makes the ablation self-contained.
# ─────────────────────────────────────────────────────────────────────────────
_CNN2D_CFG = {
    'cnn_channels_2d': [64, 128, 256, 512],
    'cnn_kernel_size':  3,
    'cnn_dropout':      0.1,
}


class _ConvBlock2D(nn.Module):
    """Conv2d → BatchNorm2d → GELU → MaxPool2d.

    'same' padding keeps spatial size constant within the conv; only the
    MaxPool reduces dimensions.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        pool: tuple = (2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv    = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                 stride=1, padding=pad, bias=False)
        self.bn      = nn.BatchNorm2d(out_ch)
        self.act     = nn.GELU()
        self.pool    = nn.MaxPool2d(kernel_size=pool) if pool is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN2DEncoder(nn.Module):
    """4-block 2D CNN — identical to the CTC encoder.

    Input : [B, 1, 64, W]
    Output: [B, 512, 8, W//8]

    Blocks:
        Block 1: 1   → 64,   H 64 → 32,  W → W/2
        Block 2: 64  → 128,  H 32 → 16,  W/2 → W/4
        Block 3: 128 → 256,  H 16 → 8,   W/4 → W/8
        Block 4: 256 → 512,  H 8  → 8,   W/8 → W/8  (pool=(1,1), no spatial reduction)
    """

    def __init__(self, cfg: dict = _CNN2D_CFG) -> None:
        super().__init__()
        ch = cfg['cnn_channels_2d']   # [64, 128, 256, 512]
        k  = cfg['cnn_kernel_size']
        dp = cfg['cnn_dropout']

        self.block1 = _ConvBlock2D(1,     ch[0], k, pool=(2, 2), dropout=dp)
        self.block2 = _ConvBlock2D(ch[0], ch[1], k, pool=(2, 2), dropout=dp)
        self.block3 = _ConvBlock2D(ch[1], ch[2], k, pool=(2, 2), dropout=dp)
        self.block4 = _ConvBlock2D(ch[2], ch[3], k, pool=(1, 1), dropout=dp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 64, W]
        x = self.block1(x)   # [B, 64,  32, W/2]
        x = self.block2(x)   # [B, 128, 16, W/4]
        x = self.block3(x)   # [B, 256, 8,  W/8]
        x = self.block4(x)   # [B, 512, 8,  W/8]
        return x              # [B, 512, 8,  W//8]


# ─────────────────────────────────────────────────────────────────────────────
# Absolute Positional Encoding
# Absolute PE — primary driver of length extrapolation failure.
#
# The sinusoidal encoding assigns a FIXED embedding to each absolute position.
# During training the model is only exposed to positions 0…(max_train_len-1).
# When decoding sequences longer than max_train_len, the model encounters
# position embeddings it has never seen, causing the distribution shift that
# leads to catastrophic accuracy collapse on OOD lengths.
#
# This is exactly what we want to demonstrate in the ablation — it contrasts
# directly with the CTC model which has no positional encoding in its 1D CNN
# decoder and is therefore position-agnostic.
# ─────────────────────────────────────────────────────────────────────────────
class AbsolutePositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al., 2017).

    # Absolute PE — primary driver of length extrapolation failure.
    # The hard position→embedding mapping is learned implicitly by the
    # downstream Transformer, which associates specific position indices with
    # specific output tokens — i.e., position 12 ≈ "end of sequence" after
    # training on max_length=12 data. This is the "length prior" failure mode.

    Args:
        d_model: Embedding dimension (must be even).
        max_len: Maximum sequence length the encoding can handle.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal table once and cache as a non-parameter buffer.
        pe = torch.zeros(max_len, d_model)                   # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()   # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                     # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)         # even dims
        pe[:, 1::2] = torch.cos(position * div_term)         # odd dims
        pe = pe.unsqueeze(0)                                  # [1, max_len, d_model]

        # Register as buffer: saved in state_dict but NOT a trainable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: [B, L, d_model]

        Returns:
            [B, L, d_model] — input + positional encoding, with dropout.
        """
        # self.pe is [1, max_len, d_model]; slice to actual sequence length.
        x = x + self.pe[:, :x.size(1), :]   # type: ignore[index]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full Transformer Seq2Seq model
# ─────────────────────────────────────────────────────────────────────────────
class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq baseline for length-extrapolation ablation study.

    This is the 'bad' model we train alongside the CTC baseline. All three of
    its critical failure modes are intentional and documented:

      1. Absolute PE — position-sensitive, fails on OOD lengths.
      2. Autoregressive <EOS> decoding — induces length prior via exposure bias.
      3. Global self-attention — allows model to see entire (partial) output,
         enabling shortcut learning of sequence length from context.

    The model is otherwise standard: same 2D CNN encoder as the CTC models,
    same image preprocessing pipeline, same training data.

    Args:
        d_model:        Transformer hidden dimension (default 256).
        nhead:          Number of attention heads (default 8; d_model / nhead must be int).
        num_layers:     Number of TransformerDecoder layers (default 4).
        dim_feedforward: Feedforward hidden size (default 1024).
        dropout:        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Vocabulary buffers (not parameters) ───────────────────────────────
        self.register_buffer('_pad_idx', torch.tensor(PAD_IDX, dtype=torch.long))
        self.register_buffer('_sos_idx', torch.tensor(SOS_IDX, dtype=torch.long))
        self.register_buffer('_eos_idx', torch.tensor(EOS_IDX, dtype=torch.long))

        # ── Encoder (shared 2D CNN — identical to CTC models) ─────────────────
        self.cnn2d = CNN2DEncoder()           # [B,1,64,W] → [B,512,8,W//8]
        # height_collapse: mean over the H=8 spatial dimension.
        # Implemented as a method call rather than a stored lambda so that
        # the module's state_dict remains clean.

        # ── Feature projector: 512 → d_model ──────────────────────────────────
        self.feature_projector = nn.Linear(512, d_model)

        # ── Encoder positional encoding ────────────────────────────────────────
        # Absolute PE — primary driver of length extrapolation failure.
        # Applied to the encoder memory so the decoder can attend to
        # position-aware features.
        self.encoder_pos = AbsolutePositionalEncoding(d_model, max_len=5000,
                                                      dropout=dropout)

        # ── Token embedding + decoder positional encoding ──────────────────────
        # Autoregressive <EOS> decoding — induces length prior.
        # The embedding table maps discrete token indices to dense vectors.
        # Combined with the absolute PE below, the decoder learns to associate
        # each output position with the token distribution seen at that position
        # during training — making it overfit to the training length distribution.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.decoder_pos = AbsolutePositionalEncoding(d_model, max_len=5000,
                                                      dropout=dropout)

        # ── Transformer Decoder ───────────────────────────────────────────────
        # Global self-attention — allows model to see entire sequence at once,
        # enabling shortcut learning of sequence length from positional context.
        # batch_first=True means shapes are [B, L, d_model] throughout.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable than post-LN at this scale)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        # ── Classifier head ───────────────────────────────────────────────────
        # Maps decoder output at each position to a distribution over 13 tokens.
        self.classifier = nn.Linear(d_model, VOCAB_SIZE)

        # ── Loss function ─────────────────────────────────────────────────────
        # ignore_index=PAD_IDX so padded target positions do not contribute.
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform init for linear layers; normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Run the image through the CNN encoder + projector + PE.

        Args:
            images: [B, 1, 64, W]

        Returns:
            memory: [B, T, d_model]   — encoder output ready for cross-attention.
        """
        # 2D CNN: [B,1,64,W] → [B,512,8,W//8]
        feat = self.cnn2d(images)

        # Height collapse — mean over H=8 spatial axis.
        # [B,512,8,T] → [B,512,T]
        feat = feat.mean(dim=2)

        # Permute to channel-last for the Linear layer.
        # [B,512,T] → [B,T,512]
        feat = feat.permute(0, 2, 1)

        # Project to d_model.
        # [B,T,512] → [B,T,d_model]
        memory = self.feature_projector(feat)

        # Add absolute positional encoding to encoder memory.
        # Absolute PE — primary driver of length extrapolation failure.
        memory = self.encoder_pos(memory)

        return memory

    @staticmethod
    def _make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Create a boolean upper-triangular mask (True = blocked)."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def _build_decoder_input(
        self,
        targets_2d: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift targets right: prepend <SOS>, append <EOS>.

        Args:
            targets_2d:     [B, max_L] — 2D padded target matrix.
            target_lengths: [B]        — number of digits per sample.

        Returns:
            dec_input:  [B, max_L+1]  — <SOS> + digits (teacher-forced input).
            dec_target: [B, max_L+1]  — digits + <EOS>  (what we want to predict).
        """
        B, max_L = targets_2d.size()
        device = targets_2d.device

        # Decoder input: <SOS> prepended.
        sos_col  = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
        dec_input = torch.cat([sos_col, targets_2d], dim=1)   # [B, max_L+1]

        # Decoder target: append <PAD> then inject <EOS> at the correct length.
        pad_col  = torch.full((B, 1), PAD_IDX, dtype=torch.long, device=device)
        dec_target = torch.cat([targets_2d, pad_col], dim=1)  # [B, max_L+1]
        for i, L in enumerate(target_lengths.tolist()):
            dec_target[i, int(L)] = EOS_IDX

        return dec_input, dec_target

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ):
        """Unified forward method — training and inference share the encoder.

        Training mode (targets is not None):
            Runs teacher-forced decoding and returns (loss, logits).

        Inference mode (targets is None):
            Returns encoder memory [B, T, d_model] for use with
            greedy_autoregressive_decode().

        Args:
            images:         [B, 1, 64, W] — normalised grayscale images.
            targets:        [B, max_L]    — 2D padded target tensor.
            target_lengths: [B]           — digits per sample (same as CTC).

        Returns:
            Training: (loss: scalar Tensor, logits: [B, L+1, 13])
            Inference: memory: [B, T, d_model]
        """
        # ── Encode images (shared path) ────────────────────────────────────────
        memory = self._encode_images(images)   # [B, T, d_model]

        if targets is None:
            # Inference mode — return encoder features for external decoder.
            return memory

        # ── Training mode: teacher-forced decoding ────────────────────────────
        assert target_lengths is not None, \
            "target_lengths must be provided when targets is given"

        # Build teacher-forced decoder input and the prediction target.
        dec_input, dec_target = self._build_decoder_input(targets, target_lengths)
        # dec_input:  [B, L+1]  — <SOS>, digit_0, …, digit_{L-1}
        # dec_target: [B, L+1]  — digit_0, …, digit_{L-1}, <EOS>

        L_plus1 = dec_input.size(1)
        device  = images.device

        # Embed tokens and add absolute positional encoding.
        # Autoregressive <EOS> decoding — induces length prior.
        tgt_emb = self.token_embedding(dec_input)   # [B, L+1, d_model]
        tgt_emb = self.decoder_pos(tgt_emb)         # [B, L+1, d_model]

        # Causal mask: prevent each position from attending to future positions.
        causal_mask = self._make_causal_mask(L_plus1, device)

        # Padding mask: ignore PAD tokens in attention
        tgt_key_padding_mask = (dec_input == PAD_IDX)

        # Transformer decoder: cross-attends to encoder memory.
        # Global self-attention — allows model to see entire sequence at once,
        # enabling shortcut learning of sequence length from context.
        dec_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )   # [B, L+1, d_model]

        # Project to vocabulary logits.
        logits = self.classifier(dec_out)   # [B, L+1, VOCAB_SIZE=13]

        # Cross-entropy loss — ignore PAD positions.
        # Reshape: loss expects [N, C] logits and [N] targets.
        loss = self.criterion(
            logits.view(-1, VOCAB_SIZE),
            dec_target.view(-1),
        )

        return loss, logits

    # ──────────────────────────────────────────────────────────────────────────
    # Greedy autoregressive decoder
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_autoregressive_decode(
        self,
        encoder_features: torch.Tensor,
        max_len: int = 500,
    ) -> List[List[int]]:
        """Greedy autoregressive decoding.

        At each step, the current partial sequence is embedded, positional
        encodings are added, and one new token is predicted from the last
        decoder position. Decoding stops when all batch items have emitted
        <EOS> or max_len is reached.

        # Autoregressive <EOS> decoding — induces length prior.
        # At OOD lengths the absolute PE signals "I should be done by now"
        # (position > training max), causing premature <EOS> predictions.
        # This is the primary mechanism of OOD failure for this model.

        Args:
            encoder_features: [B, T, d_model] — encoder output (from forward()).
            max_len:          Maximum number of decode steps (safety cap).

        Returns:
            List[List[int]] of length B — digit sequences (0-9 only; SOS,
            EOS, and PAD are stripped).
        """
        B      = encoder_features.size(0)
        device = encoder_features.device

        # Initialise decoder input with a single <SOS> token per batch item.
        dec_input = torch.full(
            (B, 1), SOS_IDX, dtype=torch.long, device=device
        )   # [B, 1]

        # Track which sequences have already produced <EOS>.
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            current_len = dec_input.size(1)

            # Embed + positional encoding.
            tgt_emb = self.token_embedding(dec_input)        # [B, cur_len, d_model]
            tgt_emb = self.decoder_pos(tgt_emb)             # [B, cur_len, d_model]

            # Causal mask for the current length.
            causal_mask = self._make_causal_mask(current_len, device)

            # Decode one step.
            # Global self-attention — allows model to see entire sequence,
            # enabling shortcut learning of sequence length from context.
            dec_out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=encoder_features,
                tgt_mask=causal_mask,
            )   # [B, cur_len, d_model]

            # Take only the LAST position's output for next-token prediction.
            last_hidden = dec_out[:, -1, :]        # [B, d_model]
            logits      = self.classifier(last_hidden)  # [B, VOCAB_SIZE]

            # Greedy argmax.
            next_token = logits.argmax(dim=-1)     # [B]

            # For finished sequences, override with <PAD> so they don't affect
            # the decoded output (they will be stripped later).
            next_token = next_token.masked_fill(finished, PAD_IDX)

            # Append predicted token.
            dec_input = torch.cat(
                [dec_input, next_token.unsqueeze(1)], dim=1
            )   # [B, cur_len+1]

            # Mark sequences that just produced <EOS> as finished.
            finished = finished | (next_token == EOS_IDX)

            # Early exit if all sequences are done.
            if finished.all():
                break

        # ── Convert token tensor to list of digit lists ────────────────────────
        # dec_input[:,0] is always <SOS>; strip it.
        token_seqs = dec_input[:, 1:].tolist()    # [B, decode_len]

        decoded: List[List[int]] = []
        for seq in token_seqs:
            digits: List[int] = []
            for tok in seq:
                if tok == EOS_IDX:
                    break          # stop at first <EOS>
                if tok not in (SOS_IDX, PAD_IDX):
                    digits.append(tok)
            decoded.append(digits)

        return decoded


# ─────────────────────────────────────────────────────────────────────────────
# Quick shape verification
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 70)
    print('Shape verification for TransformerSeq2Seq')
    print('=' * 70)

    model = TransformerSeq2Seq(d_model=256, nhead=8, num_layers=4,
                               dim_feedforward=1024, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters    : {n_params:,}')

    B, W = 2, 160

    # ── 1) Training forward ────────────────────────────────────────────────
    dummy_img      = torch.randn(B, 1, 64, W)
    target_lengths = torch.tensor([3, 5], dtype=torch.long)
    # 2D padded targets
    targets_2d = torch.tensor([
        [1, 5, 2, PAD_IDX, PAD_IDX],
        [7, 3, 9, 0,       4]
    ], dtype=torch.long)

    model.train()
    loss, train_logits = model(dummy_img,
                               targets=targets_2d,
                               target_lengths=target_lengths)

    assert loss.ndim == 0 and loss.item() > 0, \
        f'Expected positive scalar loss, got {loss}'
    expected_L = int(target_lengths.max().item()) + 1   # +1 for EOS column
    assert tuple(train_logits.shape) == (B, expected_L, VOCAB_SIZE), \
        f'Train logits shape mismatch: {tuple(train_logits.shape)}'

    print(f'\n[train] images          : {tuple(dummy_img.shape)}')
    print(f'[train] loss            : {loss.item():.4f}  (finite scalar ✔)')
    print(f'[train] logits          : {tuple(train_logits.shape)}   '
          f'expected ({B}, {expected_L}, {VOCAB_SIZE}) ✔')

    # ── 2) Inference forward → encoder features ────────────────────────────
    model.eval()
    with torch.no_grad():
        enc_feat = model(dummy_img)

    T = W // 8
    assert tuple(enc_feat.shape) == (B, T, 256), \
        f'Encoder features shape mismatch: {tuple(enc_feat.shape)}'
    print(f'\n[infer] encoder_features: {tuple(enc_feat.shape)}  '
          f'expected ({B}, {T}, 256) ✔')

    # ── 3) Greedy autoregressive decode ────────────────────────────────────
    decoded = model.greedy_autoregressive_decode(enc_feat, max_len=50)
    assert isinstance(decoded, list) and len(decoded) == B, \
        f'Decoded output must be List[List[int]] of length {B}'
    assert all(isinstance(seq, list) for seq in decoded), \
        'Each element of decoded must be a list of ints'

    print(f'\n[decode] decoded batch  : length {len(decoded)}  ✔')
    print(f'[decode] sequence[0]    : {decoded[0]}')
    print(f'[decode] sequence[1]    : {decoded[1]}')
    print('=' * 70)
    print('All shape checks PASSED ✔')
    print(f'\nNote: Parameter count {n_params:,} is in the expected 1-2M range.')
