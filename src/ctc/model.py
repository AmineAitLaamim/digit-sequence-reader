"""
CRNN + CTC model for infinite-length digit sequence reading.

Architecture:
    2D CNN  (spatial feature extraction)   -> [B, 512, 8,   W//8]
    mean over H (collapse height)         -> [B, 512,     W//8]
    1D Dilated Residual CNN (local ctx)   -> [B, 256,     W//8]
    Linear classifier                     -> [B, W//8,    11]

Why this design:
    * 2D CNN is translation-equivariant over the image — it does NOT see
      the whole image at once, so it cannot learn "if the image is very
      wide, the sequence is very long". This kills the most common shortcut
      learned by 1D-only encoders.
    * 1D Dilated CNN with dilations {1,2,4,8} has a receptive field of
      ~60 time steps. That is plenty of local context to resolve a digit
      and a separator, but it can never see the entire sequence — which
      prevents the model from learning a length prior.
    * CTC is parallel and non-autoregressive, so inference is O(T) instead
      of O(L). Crucially, CTC conditions each output frame on the input
      frame at the same time-step, not on previous outputs, which removes
      exposure-bias / error-compounding problems of autoregressive decoders.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from .config import config


# ────────────────────────────────────────────────────────────────────────────
# 2D CNN Encoder
# ────────────────────────────────────────────────────────────────────────────
class _ConvBlock2D(nn.Module):
    """Conv2d -> BatchNorm2d -> GELU -> MaxPool2d (optional)."""

    def __init__(self, in_ch, out_ch, kernel_size, pool=(2, 2)):
        super().__init__()
        # "same" padding keeps the spatial size constant within the conv,
        # so only the maxpool shrinks it.
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=1, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=pool) if pool is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=config.get('cnn_dropout', 0.1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN2DEncoder(nn.Module):
    """
    4-block 2D CNN. The first 3 blocks each halve H and W. The 4th block
    keeps spatial dims (pool=(1,1)) and only grows channels to 512.

    Input : [B, 1, 64, W]
    Output: [B, 512, 8, W//8]
    """

    def __init__(self):
        super().__init__()
        ch = config['cnn_channels_2d']   # [64, 128, 256, 512]
        k = config['cnn_kernel_size']

        # Block 1: 1  -> 64,   H 64 -> 32,  W W -> W/2
        self.block1 = _ConvBlock2D(1,                 ch[0], k, pool=(2, 2))
        # Block 2: 64 -> 128,  H 32 -> 16,  W W/2 -> W/4
        self.block2 = _ConvBlock2D(ch[0],             ch[1], k, pool=(2, 2))
        # Block 3: 128-> 256,  H 16 -> 8,   W W/4 -> W/8
        self.block3 = _ConvBlock2D(ch[1],             ch[2], k, pool=(2, 2))
        # Block 4: 256-> 512,  H 8  -> 8,   W W/8 -> W/8
        self.block4 = _ConvBlock2D(ch[2],             ch[3], k, pool=(1, 1))

    def forward(self, x):
        # x: [B, 1, 64, W]
        x = self.block1(x)   # [B, 64,  32, W/2]
        x = self.block2(x)   # [B, 128, 16, W/4]
        x = self.block3(x)   # [B, 256, 8,  W/8]
        x = self.block4(x)   # [B, 512, 8,  W/8]
        return x


# ────────────────────────────────────────────────────────────────────────────
# 1D Dilated Residual CNN
# ────────────────────────────────────────────────────────────────────────────
class ResidualConv1DBlock(nn.Module):
    """
    1D residual block:
        x -> Conv1d -> LN -> GELU -> Conv1d -> LN -> GELU -> (+residual)

    The residual is a 1x1 Conv1d projection when the channel count
    changes (first block projects 512 -> hidden_dim), otherwise identity.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation   # "same" padding for dilation

        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=kernel_size,
                               stride=1, padding=pad, dilation=dilation, bias=False)
        self.norm1 = nn.LayerNorm(out_ch)
        self.act1  = nn.GELU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size,
                               stride=1, padding=pad, dilation=dilation, bias=False)
        self.norm2 = nn.LayerNorm(out_ch)
        self.act2  = nn.GELU()

        self.dropout = nn.Dropout(p=dropout)

        # Residual projection if the channel count changes
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x: [B, C_in, T]
        residual = self.shortcut(x)

        out = self.conv1(x)
        # LayerNorm expects channel-last: [B, T, C]
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.act1(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.act2(out)

        out = self.dropout(out)
        return out + residual


class DilatedCNN1DEncoder(nn.Module):
    """
    Stack of 4 ResidualConv1DBlocks with dilations {1, 2, 4, 8}.

    Input : [B, 512, T]
    Output: [B, hidden_dim, T]
    """

    def __init__(self):
        super().__init__()
        in_ch  = 512
        out_ch = config['hidden_dim']
        k      = config['kernel_size_1d']
        dilations = config['dilations']
        dropout = config.get('resblock_dropout', 0.1)

        assert len(dilations) == config['num_res_blocks'], \
            f"len(dilations)={len(dilations)} != num_res_blocks={config['num_res_blocks']}"

        # First block: project 512 -> hidden_dim.
        # Subsequent blocks: hidden_dim -> hidden_dim (no projection needed).
        blocks = []
        for i, d in enumerate(dilations):
            blk_in  = in_ch if i == 0 else out_ch
            blk_out = out_ch
            blocks.append(ResidualConv1DBlock(blk_in, blk_out, k, d, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: [B, 512, T]
        for blk in self.blocks:
            x = blk(x)   # [B, hidden_dim, T]
        return x


# ────────────────────────────────────────────────────────────────────────────
# Full CRNN + CTC model
# ────────────────────────────────────────────────────────────────────────────
class CRNN_CTC(nn.Module):
    """
    CRNN + CTC model.

    Forward contract:
        - If `targets` is provided (training): return the scalar CTC loss.
        - If `targets` is None (inference): return raw logits
              [B, T, vocab_size]   with T = W // 8.
    """

    def __init__(self):
        super().__init__()
        self.cnn2d = CNN2DEncoder()
        self.cnn1d = DilatedCNN1DEncoder()
        self.classifier = nn.Linear(config['hidden_dim'], config['vocab_size'])

        self.ctc_loss = nn.CTCLoss(
            blank=config['BLANK_IDX'],
            zero_infinity=True,    # protects against T < L producing NaN/Inf
        )

    def forward(self, images, targets=None, target_lengths=None):
        """
        Args:
            images:         [B, 1, 64, W]    float, in [0,1] or normalised
            targets:        [N]              1D flattened long tensor of all
                                             target digits in the batch
                                             (required if computing loss)
            target_lengths: [B]              long tensor, number of digits
                                             in each item of the batch

        Returns:
            If targets is not None: scalar CTC loss
            Else:                   logits [B, T, vocab_size]
        """
        B = images.size(0)

        # ── 2D CNN ────────────────────────────────────────────────────
        feat = self.cnn2d(images)            # [B, 512, 8, T]
        T = feat.size(3)

        # ── Collapse height ───────────────────────────────────────────
        feat = feat.mean(dim=2)              # [B, 512, T]

        # ── 1D dilated CNN ────────────────────────────────────────────
        feat = self.cnn1d(feat)              # [B, hidden_dim, T]

        # ── Classifier head ───────────────────────────────────────────
        # Permute to [B, T, F] for the linear layer.
        logits = self.classifier(feat.transpose(1, 2))   # [B, T, V]

        # ── CTC training path ─────────────────────────────────────────
        if targets is not None:
            assert target_lengths is not None, \
                "target_lengths must be provided when targets is given"

            # CTC expects log-probabilities of shape [T, B, V]
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=images.device)

            loss = self.ctc_loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            # Return BOTH the loss (for backward) and the logits (for
            # metric logging) so the training loop doesn't have to do a
            # second forward pass just to compute greedy decode.
            return loss, logits

        # Inference path
        return logits


# ────────────────────────────────────────────────────────────────────────────
# Greedy CTC decoder
# ────────────────────────────────────────────────────────────────────────────
def greedy_decode(logits):
    """
    Greedy CTC decoder.

    Args:
        logits: [B, T, V]  (raw logits, NOT log-softmax)
    Returns:
        List[List[int]]   of length B, each inner list is the decoded digits
                          (BLANK tokens removed, consecutive duplicates merged).
    """
    preds = logits.argmax(dim=-1)            # [B, T]
    blank = config['BLANK_IDX']
    final_preds = []
    for pred in preds:
        collapsed = [p.item() for i, p in enumerate(pred)
                     if i == 0 or p.item() != pred[i - 1].item()]
        cleaned   = [p for p in collapsed if p != blank]
        final_preds.append(cleaned)
    return final_preds


# ────────────────────────────────────────────────────────────────────────────
# Quick shape verification
# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("Shape verification for CRNN_CTC")
    print("=" * 60)

    model = CRNN_CTC()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # ── 1) Training forward (compute loss) ────────────────────────────
    B, W = 2, 160
    dummy_img  = torch.randn(B, 1, 64, W)
    # pretend the batch has 3 and 5 digits
    target_lengths = torch.tensor([3, 5], dtype=torch.long)
    targets_flat   = torch.tensor([1, 5, 2,   7, 3, 9, 0, 4], dtype=torch.long)

    model.train()
    # forward returns (loss, logits) when targets are given.
    loss, train_logits = model(dummy_img, targets=targets_flat, target_lengths=target_lengths)
    assert tuple(train_logits.shape) == (B, W // 8, config['vocab_size']), \
        f"train logits shape mismatch: {tuple(train_logits.shape)}"
    print(f"\n[train] input image shape    : {tuple(dummy_img.shape)}")
    print(f"[train] targets shape        : {tuple(targets_flat.shape)}")
    print(f"[train] target_lengths shape : {tuple(target_lengths.shape)}")
    print(f"[train] CTC loss             : {loss.item():.4f}  (expected finite scalar)")
    print(f"[train] logits shape         : {tuple(train_logits.shape)}  (free bonus, no 2nd forward needed)")

    # ── 2) Inference forward (raw logits) ─────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(dummy_img)
    print(f"\n[infer] logits shape         : {tuple(logits.shape)}")
    expected = (B, W // 8, config['vocab_size'])
    print(f"[infer] expected shape       : {expected}")
    assert tuple(logits.shape) == expected, "Inference output shape mismatch!"
    print("[infer] ✔ shape matches [2, 20, 11]")

    # ── 3) Greedy decode ─────────────────────────────────────────────
    decoded = greedy_decode(logits)
    print(f"\n[decode] decoded batch length: {len(decoded)}")
    print(f"[decode] sample decoded seq  : {decoded[0]}  (digits only, blanks removed)")
    print("=" * 60)
    print("All shape checks PASSED ✔")
