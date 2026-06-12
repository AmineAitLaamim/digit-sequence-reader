"""
CRNN_CTC_Uncapped — Ablation model for the length-bias hypothesis.

Architecture is identical to CRNN_CTC (src/ctc/model.py) EXCEPT for the 1D
Dilated Residual CNN, which uses 6 blocks with dilations [1, 2, 4, 8, 16, 32]
instead of 4 blocks with [1, 2, 4, 8].

Receptive field comparison
──────────────────────────
  Original  (4 blocks, d=[1,2,4,8])    : RF ≈ 2*Σ(k-1)*d = 2*(2+4+8+16) = 60 steps
  Uncapped  (6 blocks, d=[1,2,4,8,16,32]): RF ≈ 2*(2+4+8+16+32+64)     = 252 steps

At 252 steps the model can easily see a training sequence of ≤12 digits in its
entirety (each digit takes roughly 8 steps after the 8× CNN downsampling).
This lets the model learn a "global length prior" — e.g., counting the total
number of digits from the full image — which is precisely the shortcut we want
to prove is harmful for length generalisation.

Expected outcome of the ablation
─────────────────────────────────
  * The uncapped model should perform equally well or better on in-distribution
    sequences (trained with max_length=12).
  * On out-of-distribution sequences (length > 12) its accuracy should degrade
    significantly compared to the original CRNN_CTC, demonstrating that
    capping the receptive field is necessary for length generalisation.

Architecture overview
─────────────────────
  2D CNN Encoder         [B, 1, 64, W]   → [B, 512, 8, W//8]
  Height Collapse        mean(dim=2)      → [B, 512,    W//8]
  1D Dilated Res-CNN     6 blocks         → [B, 256,    W//8]   ← CHANGED
  Linear classifier                       → [B, W//8,  11]
  CTCLoss (blank=10)
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (self-contained — no dependency on src.ctc.config so that
# this module can be imported stand-alone and the ablation config is explicit).
# ─────────────────────────────────────────────────────────────────────────────
_UNCAPPED_CONFIG = {
    # 2D CNN (identical to original)
    'cnn_channels_2d': [64, 128, 256, 512],
    'cnn_kernel_size':  3,
    'cnn_dropout':      0.1,

    # 1D Dilated CNN — ABLATION: 6 blocks, dilations [1,2,4,8,16,32]
    # Receptive field ≈ 252 steps (vs. 60 in the original capped model).
    'hidden_dim':       256,
    'dilations':        [1, 2, 4, 8, 16, 32],   # UNCAPPED — see module docstring
    'kernel_size_1d':   3,
    'resblock_dropout': 0.1,

    # Vocabulary — 10 digits (0-9) + 1 CTC BLANK
    'vocab_size':  11,
    'BLANK_IDX':   10,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2D CNN Encoder  (identical to src/ctc/model.py — copied verbatim for
# isolation so the ablation does not create a hard import dependency on ctc/)
# ─────────────────────────────────────────────────────────────────────────────
class _ConvBlock2D(nn.Module):
    """Conv2d → BatchNorm2d → GELU → MaxPool2d.

    'same' padding keeps spatial size constant within the conv; only the
    MaxPool reduces dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 pool: tuple = (2, 2), dropout: float = 0.1):
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
    """4-block 2D CNN — identical to the original CRNN_CTC encoder.

    Input : [B, 1, 64, W]
    Output: [B, 512, 8, W//8]

    Blocks:
        Block 1: 1   → 64,   H 64 → 32,  W → W/2
        Block 2: 64  → 128,  H 32 → 16,  W/2 → W/4
        Block 3: 128 → 256,  H 16 → 8,   W/4 → W/8
        Block 4: 256 → 512,  H 8  → 8,   W/8 → W/8  (pool=(1,1), no-op pool)
    """

    def __init__(self, cfg: dict):
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
# 1D Dilated Residual CNN  (THE CRITICAL CHANGE)
# ─────────────────────────────────────────────────────────────────────────────
class ResidualConv1DBlock(nn.Module):
    """1D residual block with dilation.

    Forward:
        x → Conv1d → LN → GELU → Conv1d → LN → GELU → (+ residual)

    The residual is a learned 1×1 projection when in_ch ≠ out_ch (first block
    projects 512 → hidden_dim); for all later blocks it is nn.Identity().

    LayerNorm is applied channel-last (transpose trick) to be consistent with
    the original implementation.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        # 'same' padding for the given dilation:
        #   pad = (k - 1) // 2 * dilation
        pad = (kernel_size - 1) // 2 * dilation

        self.conv1   = nn.Conv1d(in_ch,  out_ch, kernel_size=kernel_size,
                                 stride=1, padding=pad, dilation=dilation, bias=False)
        self.norm1   = nn.LayerNorm(out_ch)
        self.act1    = nn.GELU()

        self.conv2   = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size,
                                 stride=1, padding=pad, dilation=dilation, bias=False)
        self.norm2   = nn.LayerNorm(out_ch)
        self.act2    = nn.GELU()

        self.dropout = nn.Dropout(p=dropout)

        # Residual projection only when channel dims differ.
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        residual = self.shortcut(x)         # [B, C_out, T]

        out = self.conv1(x)                 # [B, C_out, T]
        # LayerNorm expects [B, T, C] — transpose, norm, transpose back.
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act1(out)

        out = self.conv2(out)               # [B, C_out, T]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act2(out)

        out = self.dropout(out)
        return out + residual               # [B, C_out, T]


class UncappedDilatedCNN1DEncoder(nn.Module):
    """
    Ablation: Uncapped receptive field to test length bias hypothesis.

    Stack of 6 ResidualConv1DBlocks with dilations [1, 2, 4, 8, 16, 32].

    Receptive field:
        RF ≈ 2 × Σ_{d ∈ dilations} (k-1) × d
           = 2 × (2 + 4 + 8 + 16 + 32 + 64)
           = 252 time steps

    At 252 steps the model can see entire training sequences (≤12 digits ≈
    ≤96 steps after 8× downsampling) in a single receptive window.  This
    enables the pathological shortcut of learning a global length prior, which
    we hypothesise causes catastrophic failure on out-of-distribution lengths.

    Input : [B, 512, T]   (T = W // 8)
    Output: [B, 256, T]
    """

    def __init__(self, cfg: dict):
        super().__init__()
        in_ch   = 512
        out_ch  = cfg['hidden_dim']        # 256
        k       = cfg['kernel_size_1d']    # 3
        # Ablation: Uncapped receptive field to test length bias hypothesis.
        # Dilations [1, 2, 4, 8, 16, 32] → RF ≈ 252 steps.
        dilations = cfg['dilations']       # [1, 2, 4, 8, 16, 32]
        dropout = cfg['resblock_dropout']

        blocks = []
        for i, d in enumerate(dilations):
            blk_in  = in_ch if i == 0 else out_ch   # 512 → 256 on block 0
            blk_out = out_ch
            blocks.append(ResidualConv1DBlock(blk_in, blk_out, k, d, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512, T]
        for blk in self.blocks:
            x = blk(x)   # [B, 256, T]
        return x          # [B, 256, T]


# ─────────────────────────────────────────────────────────────────────────────
# Full CRNN_CTC_Uncapped model
# ─────────────────────────────────────────────────────────────────────────────
class CRNN_CTC_Uncapped(nn.Module):
    """
    CRNN + CTC with an **uncapped** 1D dilated receptive field.

    This is an ablation model: the only architectural difference from the
    production CRNN_CTC is the 1D encoder, which uses 6 dilated blocks
    (dilations [1,2,4,8,16,32], RF ≈ 252) instead of 4 (dilations [1,2,4,8],
    RF ≈ 60).  All other components — 2D CNN, height collapse, classifier head,
    and CTC loss — are identical.

    Forward contract (same as CRNN_CTC):
        - targets provided  → returns (scalar CTC loss, logits [B, T, V])
        - targets is None   → returns logits [B, T, V]   (inference)

    Tensor shapes at each stage:
        Input images          : [B, 1, 64, W]
        After CNN2DEncoder    : [B, 512, 8, W//8]
        After height collapse : [B, 512,    W//8]
        After 1D dilated CNN  : [B, 256,    W//8]
        Logits                : [B, W//8,   11]
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        cfg = cfg or _UNCAPPED_CONFIG

        self.cnn2d      = CNN2DEncoder(cfg)
        self.cnn1d      = UncappedDilatedCNN1DEncoder(cfg)
        self.classifier = nn.Linear(cfg['hidden_dim'], cfg['vocab_size'])

        self.ctc_loss = nn.CTCLoss(
            blank=cfg['BLANK_IDX'],
            zero_infinity=True,   # guards against T < L producing NaN / Inf
        )
        self._blank_idx = cfg['BLANK_IDX']

    def forward(
        self,
        images:         torch.Tensor,
        targets:        torch.Tensor = None,
        target_lengths: torch.Tensor = None,
    ):
        """
        Args:
            images:         [B, 1, 64, W]   float, in [0,1] or normalised
            targets:        [N]             1D flattened long tensor of all
                                            target digits in the batch
                                            (required when computing loss)
            target_lengths: [B]             long tensor — number of digits per item

        Returns:
            (loss, logits [B, T, V])  if targets is not None
            logits [B, T, V]          if targets is None  (inference)
        """
        B = images.size(0)

        # ── 2D CNN ────────────────────────────────────────────────────────
        feat = self.cnn2d(images)           # [B, 512, 8, T]  T = W // 8
        T    = feat.size(3)

        # ── Height collapse ───────────────────────────────────────────────
        feat = feat.mean(dim=2)             # [B, 512, T]

        # ── 1D Dilated CNN (UNCAPPED — RF ≈ 252 steps) ───────────────────
        feat = self.cnn1d(feat)             # [B, 256, T]

        # ── Classifier head ───────────────────────────────────────────────
        # Permute to [B, T, F] for the Linear layer.
        logits = self.classifier(feat.transpose(1, 2))   # [B, T, 11]

        # ── CTC training path ─────────────────────────────────────────────
        if targets is not None:
            assert target_lengths is not None, \
                "target_lengths must be provided when targets is given"

            # CTCLoss requires log-probabilities of shape [T, B, V].
            log_probs    = logits.log_softmax(dim=-1).transpose(0, 1)   # [T, B, 11]
            input_lengths = torch.full((B,), T, dtype=torch.long, device=images.device)

            loss = self.ctc_loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            # Return BOTH loss and logits so the training loop can log metrics
            # without a second forward pass.
            return loss, logits   # scalar, [B, T, 11]

        # ── Inference path ────────────────────────────────────────────────
        return logits             # [B, T, 11]


# ─────────────────────────────────────────────────────────────────────────────
# Greedy CTC decoder (self-contained copy — no import from src.ctc.model)
# ─────────────────────────────────────────────────────────────────────────────
def greedy_decode(logits: torch.Tensor, blank: int = 10):
    """
    Greedy CTC decoder.

    Args:
        logits: [B, T, V]  raw logits (NOT log-softmax)
        blank:  int        index of the CTC BLANK token (default 10)

    Returns:
        List[List[int]]  — length B, each inner list is the decoded digit
                           sequence (blanks removed, consecutive duplicates merged).
    """
    preds       = logits.argmax(dim=-1)   # [B, T]
    final_preds = []
    for pred in preds:
        # Collapse consecutive duplicates, then strip blanks.
        collapsed = [
            p.item() for i, p in enumerate(pred)
            if i == 0 or p.item() != pred[i - 1].item()
        ]
        cleaned = [p for p in collapsed if p != blank]
        final_preds.append(cleaned)
    return final_preds


# ─────────────────────────────────────────────────────────────────────────────
# Quick shape verification
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("Shape verification for CRNN_CTC_Uncapped")
    print("=" * 65)

    model   = CRNN_CTC_Uncapped()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {n_params:,}")
    print(f"Dilations        : {_UNCAPPED_CONFIG['dilations']}")
    print(f"Receptive field  : ~252 time steps (vs. ~60 for original CRNN_CTC)")

    B, W = 2, 160

    # 1) Training forward
    dummy_img      = torch.randn(B, 1, 64, W)
    target_lengths = torch.tensor([3, 5], dtype=torch.long)
    targets_flat   = torch.tensor([1, 5, 2, 7, 3, 9, 0, 4], dtype=torch.long)

    model.train()
    loss, train_logits = model(dummy_img, targets=targets_flat,
                                target_lengths=target_lengths)
    assert tuple(train_logits.shape) == (B, W // 8, 11), \
        f"train logits shape mismatch: {tuple(train_logits.shape)}"
    print(f"\n[train] image shape        : {tuple(dummy_img.shape)}")
    print(f"[train] CTC loss           : {loss.item():.4f}  (finite scalar [OK])")
    print(f"[train] logits shape       : {tuple(train_logits.shape)}")

    # 2) Inference forward
    model.eval()
    with torch.no_grad():
        logits = model(dummy_img)
    expected = (B, W // 8, 11)
    assert tuple(logits.shape) == expected, \
        f"Inference shape mismatch: {tuple(logits.shape)} vs {expected}"
    print(f"\n[infer] logits shape       : {tuple(logits.shape)}")
    print(f"[infer] expected shape     : {expected}  [OK]")

    # 3) Greedy decode
    decoded = greedy_decode(logits)
    print(f"\n[decode] batch length      : {len(decoded)}")
    print(f"[decode] sample sequence   : {decoded[0]}")
    print("=" * 65)
    print("All shape checks PASSED [OK]")
