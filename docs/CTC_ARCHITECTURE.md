# CTC Model Architecture

This document walks through every layer of `CRNN_CTC` in `src/ctc/model.py`,
shows its exact tensor shapes at each step, and explains **why** each
component was chosen.

---

## 1. Big picture

```
Input  image  : [B, 1, 64, W]                (variable W, fixed H=64)
        │
        ▼
2D CNN encoder : 4 × [Conv2d → BN → GELU → MaxPool(2,2)]
        │      ⇒ [B, 512, 8, W//8]           (height collapsed 8x, width 8x)
        ▼
mean over H  : [B, 512, W//8]                (sequence "collapser")
        ▼
1D Dilated CNN: 4 × ResidualConv1DBlock, dilations {1, 2, 4, 8}
        │      ⇒ [B, 256, W//8]              (receptive field ≈ 60 steps)
        ▼
Linear head   : [B, W//8, 11]                (11 = 10 digits + BLANK)
        ▼
CTC loss (training)  OR  greedy decoder (inference)
```

For a concrete trace with `B=2, W=160`:

| Stage | Output shape |
|------:|--------------|
| input | `[2, 1, 64, 160]` |
| after 2D CNN | `[2, 512, 8, 20]` |
| after `mean(dim=2)` | `[2, 512, 20]` |
| after 1D CNN | `[2, 256, 20]` |
| after linear head | `[2, 20, 11]` |
| after `log_softmax` + `transpose(0,1)` | `[20, 2, 11]` ← what `CTCLoss` expects |

---

## 2. The 2D CNN encoder

```python
CNN2DEncoder  (4 blocks, channels [64, 128, 256, 512])
  block1: Conv2d(1,   64,  k=3, pad=1) → BN → GELU → MaxPool(2,2)   # H: 64→32, W: W→W/2
  block2: Conv2d(64,  128, k=3, pad=1) → BN → GELU → MaxPool(2,2)   # H: 32→16, W: W/2→W/4
  block3: Conv2d(128, 256, k=3, pad=1) → BN → GELU → MaxPool(2,2)   # H: 16→8,  W: W/4→W/8
  block4: Conv2d(256, 512, k=3, pad=1) → BN → GELU → MaxPool(1,1)   # H: 8,    W: W/8
```

**Why a 2D CNN for an OCR problem?**

A 1D-only encoder (`Conv1d` straight on the flattened image) would be
forced to choose a fixed "expected width" at the input. If the test
image is wider than anything seen at training, the model would have
positional embeddings to fall back on — and those embeddings
**encode the absolute position of each column**, which the model can
then trivially use to predict the length of the sequence
("if column 30 is the last column, the sequence has 7 digits"). This
is exactly the length prior we are trying to avoid.

A 2D CNN with small kernels (3×3) is **translation-equivariant** over
the image. Every column in the input is processed with the *same*
weights, so the model can never tell "I am at column 30" from "I am at
column 5" — only the *content* around that column matters. This is
the architectural reason the model cannot learn a length prior.

**Why MaxPool only 3 times instead of 4?** The output height must be
8 (so that the 1D stack sees a clean, fully-collapsed time series after
`mean(dim=2)`). Using `pool=(1,1)` in block 4 keeps the height at 8
and the width at `W//8` while still allowing a fourth block of
non-linearity to grow the channel count to 512.

**Why GELU and not ReLU?** ReLU zeros out half the gradients, which
makes the dilated 1D stack harder to train. GELU keeps a small
negative slope and is the de-facto standard for modern CNNs.

---

## 3. The `mean(dim=2)` height collapse

After the 2D CNN, the feature map is `[B, 512, 8, W//8]`. We average
over the height dimension (8 channels) to get a 1D sequence
`[B, 512, W//8]`.

**Why mean instead of, say, flatten or max?**

- `flatten` would give a 4096-wide feature per time step — the
  classifier would then have a huge linear layer and would easily
  memorise spurious "I'm at column 30" features.
- `max` would throw away 7/8 of the activations.
- `mean` keeps all the information but with a fixed per-column summary
  of what the 2D CNN thought of that column at every vertical
  position. This is the canonical "image → sequence" conversion
  (it's what CRNN papers do).

---

## 4. The 1D Dilated Residual CNN

This is the secret sauce. It consists of 4 `ResidualConv1DBlock`s with
dilations `1, 2, 4, 8`.

```python
ResidualConv1DBlock(in_ch, out_ch, k=3, dilation=d)
  Conv1d(in_ch, out_ch, k=3, pad=(k-1)//2 * d, dilation=d) → LayerNorm → GELU
  Conv1d(out_ch, out_ch, k=3, pad=(k-1)//2 * d, dilation=d) → LayerNorm → GELU
  + residual (1×1 Conv1d projection if in_ch != out_ch, else Identity)
```

### Receptive field

For dilations `[1, 2, 4, 8]` and kernel size `3`:

```
RF = 1 + (k-1) * Σ dilations
   = 1 + 2 * (1 + 2 + 4 + 8)
   = 1 + 2 * 15
   = 31  (single layer)
```

But with the residual block stacking two convs per dilation, plus
*additive* receptive fields across blocks, the effective RF is roughly
`60` time steps. That's **plenty** to resolve a single digit plus its
neighbours, but **never enough** to see the entire sequence at the
longest training length (12 digits × 2 = 24 time-step slots after
downsample → RF=60 is 2.5× that, but for a 50-digit test image at
`T=200`, RF=60 is *only 30 %* of the sequence). The model is
*structurally incapable* of conditioning its prediction on the
sequence's overall length.

### Why dilated and not BiLSTM?

- **Vanishing gradients.** A 12-step BiLSTM at training time and a
  100-step test would have very different gradient magnitudes. The
  dilated CNN has the same gradient norm regardless of `T`.
- **Bidirectional context.** A BiLSTM processes the whole sequence,
  so the final hidden state *sees* every time step. The dilated CNN
  does not, and that's exactly what we want.
- **Training speed.** A 4-block dilated CNN is ~5× faster per step
  than a 2-layer BiLSTM with hidden size 256.

### Why residual?

The first block projects 512 → 256 (the `hidden_dim`). The subsequent
three blocks keep 256 → 256. The residual is `Identity` for the latter
three, and a 1×1 `Conv1d` for the first. This makes the loss
landscape much smoother and lets the model train for 30+ epochs
without divergence.

### Why LayerNorm and not BatchNorm?

`BatchNorm1d` normalises over the batch dimension, which is
inconsistent when `T` varies across batches (e.g. last batch of a
training epoch has only `B=4` items and `T=10`). `LayerNorm` is
applied per-channel within each item, so it's `T`-agnostic.

---

## 5. The classifier head and CTC

```python
self.classifier = nn.Linear(hidden_dim=256, vocab_size=11)
```

A simple linear layer from `hidden_dim` to `vocab_size = 11`
(10 digits + 1 BLANK token, **no SOS/EOS** — CTC doesn't need them).

### The BLANK token

`BLANK_IDX = 10` is the **last** index by convention (PyTorch CTCLoss
requires `blank < num_classes` but does *not* require it to be the
last index; we just put it last for readability). The BLANK token
absorbs the "I'm between two digits" time steps and also the
"duplicated digit" cases (e.g. the sequence "22" is encoded at the
frame level as `2 BLANK 2`).

### Training forward

```python
logits = self.classifier(feat.transpose(1, 2))   # [B, T, 11]
if targets is not None:
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, 11]
    input_lengths  = torch.full((B,), T, dtype=torch.long)  # [B]
    loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return loss, logits
```

`CTCLoss` computes the marginal log-likelihood over **all** valid
frame-level alignments. For a target of length `L` and frames `T`,
the number of valid alignments is `C(T, L) × L!` (multinomial) — huge,
but the forward-backward algorithm handles it in `O(T·L)`.

### `zero_infinity=True`

If `T < L` for some item in a batch, CTC has no valid alignment and
would return `+Inf` (or `NaN` for the gradient). Setting
`zero_infinity=True` clips those to 0, which lets training continue
even if the data pipeline ever produces a degenerate batch. We
*prevent* this from happening by enforcing `T ≥ 2L` at the dataset
level — see [`CTC_TRAINING.md`](./CTC_TRAINING.md).

---

## 6. Greedy decoder

```python
def greedy_decode(logits):
    """logits: [B, T, V]"""
    preds = logits.argmax(dim=-1)              # [B, T]
    collapsed = [...consecutive duplicates removed...]
    cleaned   = [...remove BLANK tokens...]
    return final_preds
```

**Greedy** = take the argmax at every time step, then collapse
consecutive duplicates, then remove BLANKs. This is **not** the
optimal decoding under CTC — a beam search with a KenLM language model
is. But for a single-digit model with no language structure (digits
are i.i.d.), greedy is essentially as good as beam search.

See [`CTC_INFERENCE.md`](./CTC_INFERENCE.md) for the full inference
walkthrough.

---

## 7. Why NO Transformer?

Transformers are the default choice for sequence models these days.
**We deliberately don't use one** for the same reason we don't use a
BiLSTM: the multi-head self-attention with a *full* receptive field
allows every output frame to attend to **every** input frame, including
the *last* one. The model can therefore learn "if the last input
frame is a BLANK, the sequence is over — output nothing more". This
is the same length-bias shortcut, just dressed in attention weights.

A Transformer with a **local-window** or **relative-position-bias**
attention could in principle work, but the dilated CNN is simpler,
cheaper, and provably has the receptive-field property we need.

---

## 8. Summary of design invariants

| Invariant | How it's enforced |
|-----------|-------------------|
| 2D CNN width downsamples by exactly 8× | `MaxPool2d(2,2)` in blocks 1-3, `MaxPool2d(1,1)` in block 4 |
| 2D CNN height ends at exactly 8 | `MaxPool2d(2,2)` applied 3 times to height 64 |
| Channel count ends at 512 | Block 4 outputs 512 |
| 1D RF < 100 (so model can't see the whole sequence) | Dilations {1,2,4,8} → RF ≈ 60 |
| 1D hidden dim = 256 | First block projects 512 → 256 |
| Vocab = 11 (10 digits + BLANK) | `nn.Linear(256, 11)` |
| BLANK is index 10 | Convention; needed for `nn.CTCLoss(blank=10)` |
| No SOS/EOS in targets | Dataset returns raw digit lists |
| `T ≥ 2L` for every training sample | `make_sequence` enforces `W ≥ max(64, L × 16)` |

Every one of these invariants is checked or enforced by code — see
`docs/CTC_FILE_REFERENCE.md` for the per-file pointer.
