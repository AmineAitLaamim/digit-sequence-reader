# CRNN_CTC_Uncapped — Ablation Study

This document describes the **ablation variant** of the production `CRNN_CTC`
model, explains the scientific hypothesis it tests, walks through the one
architectural change made, and shows how to reproduce the experiment.

> **TL;DR** — We remove the receptive-field cap from the 1D encoder (4 blocks
> with dilations `[1,2,4,8]`, RF ≈ 60 steps) and replace it with 6 blocks
> using dilations `[1,2,4,8,16,32]` (RF ≈ 252 steps). The 2D CNN, height
> collapse, classifier head, and CTC loss are **identical** to the baseline.

---

## 1. The Hypothesis

The baseline `CRNN_CTC` is designed around one core claim:

> *A model that cannot see the entire input sequence in a single receptive
> window cannot learn a length prior — and therefore generalises to sequence
> lengths far beyond its training distribution.*

The two architectural choices that enforce this are:
1. The **2D CNN** — translation-equivariant, never sees the full width.
2. The **1D Dilated CNN with RF ≈ 60** — enough local context for a digit,
   not enough to see the whole sequence.

**The ablation asks:** if we break constraint 2 by making the receptive field
large enough to cover an entire training sequence, does length generalisation
fail?

If yes → the RF cap is *necessary*, not just a nice-to-have.

---

## 2. The One Architectural Change

### Baseline (`src/ctc/model.py`)

```
1D Dilated Residual CNN
  Block 0: dilation=1   ─┐
  Block 1: dilation=2    │  4 blocks
  Block 2: dilation=4    │  RF ≈ 60 time steps
  Block 3: dilation=8   ─┘
```

### Ablation (`src/CRNN_CTC_Uncapped/model.py`)

```
1D Dilated Residual CNN  (UNCAPPED)
  Block 0: dilation=1   ─┐
  Block 1: dilation=2    │
  Block 2: dilation=4    │  6 blocks
  Block 3: dilation=8    │  RF ≈ 252 time steps
  Block 4: dilation=16   │
  Block 5: dilation=32  ─┘
```

Everything else — `CNN2DEncoder`, `mean(dim=2)`, `nn.Linear(256, 11)`,
`nn.CTCLoss(blank=10, zero_infinity=True)` — is bit-for-bit identical.

---

## 3. Receptive Field Calculation

For a stack of residual blocks with kernel size `k=3`:

```
RF ≈ 2 × Σ (k-1) × d   for each dilation d in the stack

Baseline  : 2 × (2 + 4 + 8 + 16)             =  60 steps
Uncapped  : 2 × (2 + 4 + 8 + 16 + 32 + 64)   = 252 steps
```

At 8× CNN downsampling, a 12-digit training sequence occupies at most
`12 × 16 / 8 = 24` time steps. RF = 252 covers **10× that** — the model
can comfortably see the entire training sequence in one window and learn
to count digits globally.

---

## 4. File Structure

```
src/
├── ctc/                         # Baseline (production) model
│   ├── config.py
│   ├── model.py                 # CRNN_CTC, greedy_decode
│   ├── dataset.py               # InfiniteCTCDataset, collate_fn
│   ├── train.py
│   └── ...
└── CRNN_CTC_Uncapped/           # Ablation model (this document)
    ├── __init__.py
    └── model.py                 # CRNN_CTC_Uncapped, greedy_decode

notebooks/
└── train_colab_ctc_uncapped.ipynb   # Colab training notebook
```

### Why the ablation model is self-contained

`src/CRNN_CTC_Uncapped/model.py` does **not** import from `src/ctc/config.py`.
It carries its own `_UNCAPPED_CONFIG` dict with the hyperparameters written
out explicitly. This serves two purposes:

1. **Isolation** — a future change to the baseline config cannot silently
   affect the ablation results.
2. **Readability** — the precise configuration of the ablation is visible in
   one place without chasing imports.

The dataset and training utilities, however, **are** reused from `src/ctc/`
(see the notebook) to guarantee the data pipeline is byte-for-byte identical.

---

## 5. Tensor Shapes at Each Stage

For `B=2, W=160` (the shape-check scenario):

| Stage | Shape |
|-------|-------|
| Input image | `[B, 1, 64, W]` → `[2, 1, 64, 160]` |
| After `CNN2DEncoder` | `[B, 512, 8, W//8]` → `[2, 512, 8, 20]` |
| After `mean(dim=2)` | `[B, 512, W//8]` → `[2, 512, 20]` |
| After `UncappedDilatedCNN1DEncoder` | `[B, 256, W//8]` → `[2, 256, 20]` |
| After `classifier` | `[B, T, 11]` → `[2, 20, 11]` |
| `log_softmax + transpose(0,1)` | `[T, B, 11]` → `[20, 2, 11]` ← `CTCLoss` input |

These are asserted by the built-in `__main__` check (see §7).

---

## 6. Block-by-Block Architecture

### 6A. `CNN2DEncoder` — unchanged

```python
Block 1: Conv2d(1,   64,  k=3, pad=1) → BN → GELU → MaxPool(2,2)  # H: 64→32, W: W→W/2
Block 2: Conv2d(64,  128, k=3, pad=1) → BN → GELU → MaxPool(2,2)  # H: 32→16, W: W/2→W/4
Block 3: Conv2d(128, 256, k=3, pad=1) → BN → GELU → MaxPool(2,2)  # H: 16→8,  W: W/4→W/8
Block 4: Conv2d(256, 512, k=3, pad=1) → BN → GELU → MaxPool(1,1)  # H: 8,     W: W/8
```

### 6B. `UncappedDilatedCNN1DEncoder` — the critical change

Each of the 6 `ResidualConv1DBlock`s has the same internal structure:

```python
ResidualConv1DBlock(in_ch, out_ch, k=3, dilation=d)
  pad = (k-1)//2 * d                          # "same" padding for this dilation
  Conv1d(in_ch,  out_ch, k=3, pad, dilation=d) → LayerNorm → GELU
  Conv1d(out_ch, out_ch, k=3, pad, dilation=d) → LayerNorm → GELU
  + shortcut (1×1 Conv1d if in_ch != out_ch, else Identity)
```

| Block | `in_ch` | `out_ch` | Dilation | Shortcut |
|------:|--------:|---------:|---------:|----------|
| 0 | 512 | 256 | 1 | 1×1 Conv1d (channel projection) |
| 1 | 256 | 256 | 2 | Identity |
| 2 | 256 | 256 | 4 | Identity |
| 3 | 256 | 256 | 8 | Identity |
| 4 | 256 | 256 | 16 | Identity |
| 5 | 256 | 256 | 32 | Identity |

### 6C. Classifier head and CTC — unchanged

```python
self.classifier = nn.Linear(256, 11)          # 10 digits + BLANK
self.ctc_loss   = nn.CTCLoss(blank=10, zero_infinity=True)
```

---

## 7. Running the Shape Sanity-Check Locally

```bash
# From the repo root, using your project virtualenv
python -m src.CRNN_CTC_Uncapped.model
```

Expected output:

```
=================================================================
Shape verification for CRNN_CTC_Uncapped
=================================================================
Total parameters : 4,246,731
Dilations        : [1, 2, 4, 8, 16, 32]
Receptive field  : ~252 time steps (vs. ~60 for original CRNN_CTC)

[train] image shape        : (2, 1, 64, 160)
[train] CTC loss           : <finite scalar>  (finite scalar [OK])
[train] logits shape       : (2, 20, 11)

[infer] logits shape       : (2, 20, 11)
[infer] expected shape     : (2, 20, 11)  [OK]

[decode] batch length      : 2
[decode] sample sequence   : [...]
=================================================================
All shape checks PASSED [OK]
```

---

## 8. Training the Ablation on Colab

Open [`notebooks/train_colab_ctc_uncapped.ipynb`](../notebooks/train_colab_ctc_uncapped.ipynb).

The notebook covers:

| Cell | What it does |
|------|-------------|
| 1 · Drive mount | Mounts Google Drive; creates `checkpoint_CRNN_CTC_Uncapped/` |
| 2 · Clone & install | Clones the repo and installs requirements |
| 3 · Verify GPU | Checks CUDA is available |
| 4 · Path setup & imports | Imports `CRNN_CTC_Uncapped` and reuses `src/ctc/` dataset utilities |
| 5 · Hyperparameters | Sets `max_length=12` (same cap as baseline — critical for fair comparison) |
| 6 · Sanity check | Runs `python -m src.CRNN_CTC_Uncapped.model` |
| 7 · Build data loaders | Calls existing `get_dataloaders()` from `src/ctc/dataset.py` |
| 8 · Model initialisation | Instantiates `CRNN_CTC_Uncapped`, AdamW, `ReduceLROnPlateau` |
| 9 · Helper functions | Levenshtein CER, `validate()`, `save_checkpoint()` |
| 10 · Training loop | Full loop with grad clipping, early stopping, Drive checkpoint saving |
| 11 · Training curves | `matplotlib` loss / seq-acc / char-acc plots |
| 12 · **Length-extrapolation test** | Evaluates at `L ∈ {3,5,7,10,12,15,20,25,30}` — the key ablation result |
| 13 · Custom inference | Upload your own image for inference |

### Key training settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| `max_length` | 12 | Same as baseline — ensures identical in-distribution training |
| `batch_size` | 64 | Same as baseline |
| `lr` | 1e-3 | Same as baseline |
| `clip_grad` | 1.0 | Same as baseline |
| `CHECKPOINT_DIR` | `/content/drive/MyDrive/checkpoint_CRNN_CTC_Uncapped` | Separate from baseline so checkpoints don't overwrite each other |
| Checkpoint file | `best_ctc_uncapped.pt` | Named differently for clarity |

---

## 9. Expected Experimental Outcome

### In-distribution performance (`L ≤ 12`)

Both models should achieve similar sequence accuracy and CER on sequences
within the training distribution. If the uncapped model is *worse* here,
the extra blocks are causing overfitting or gradient issues — investigate
the training curves.

### Out-of-distribution performance (`L > 12`) — the ablation result

| Sequence length | Baseline `CRNN_CTC` (RF≈60) | `CRNN_CTC_Uncapped` (RF≈252) | Interpretation |
|----------------:|:----------------------------:|:------------------------------:|----------------|
| 7 (in-dist) | ~95 % | ~95 % | Both should match |
| 12 (in-dist) | ~95 % | ~95 % | Both should match |
| 15 (OOD ×1.3) | ~85 % | < baseline | Uncapped starts to fail |
| 20 (OOD ×1.7) | ~70 % | ≪ baseline | Failure mode visible |
| 30 (OOD ×2.5) | ~50 % | ≪ baseline | Catastrophic degradation |

> **Note:** The numbers above are illustrative. Run cell 12 of the notebook
> to obtain the actual curves for your trained models.

If the hypothesis holds, the OOD gap between the two models confirms that
**capping the receptive field is necessary for length generalisation**, and
that the 252-step window allows the model to learn a global length prior
that breaks at test time.

---

## 10. Parameter Count Comparison

| Model | 1D blocks | Dilations | Approx. params |
|-------|----------:|-----------|---------------:|
| `CRNN_CTC` (baseline) | 4 | `[1,2,4,8]` | ~3.4 M |
| `CRNN_CTC_Uncapped` | 6 | `[1,2,4,8,16,32]` | ~4.2 M |

The two extra blocks add approximately **800 K parameters**. This is a
moderate increase and is not the intended explanation for any performance
difference. The extra capacity could even *help* in-distribution — the
key metric is the out-of-distribution degradation.

---

## 11. DRY Principle — What Is and Isn't Shared

| Component | Shared with baseline? | Where |
|-----------|----------------------|-------|
| `CNN2DEncoder` | Copied (isolated) | `src/CRNN_CTC_Uncapped/model.py` |
| `ResidualConv1DBlock` | Copied (isolated) | `src/CRNN_CTC_Uncapped/model.py` |
| `greedy_decode` | Copied (isolated) | `src/CRNN_CTC_Uncapped/model.py` |
| `InfiniteCTCDataset` | **Reused directly** | `from src.ctc.dataset import ...` |
| `collate_fn` | **Reused directly** | `from src.ctc.dataset import ...` |
| `get_dataloaders` | **Reused directly** | `from src.ctc.dataset import ...` |
| `config` dict | **Reused directly** | `from src.ctc.config import config` |

The model is isolated to avoid contamination of the ablation by future
baseline changes. The data pipeline is shared to guarantee the experiment
is a controlled, single-variable test.

---

## 12. Relation to Other Documentation

| Document | Relation |
|----------|----------|
| [`CTC_ARCHITECTURE.md`](./CTC_ARCHITECTURE.md) | Full description of the baseline model this ablation is derived from |
| [`CTC_EXTRAPOLATION.md`](./CTC_EXTRAPOLATION.md) | Theoretical argument for why a capped RF prevents length-bias; this ablation provides the empirical counterpart |
| [`CTC_TRAINING.md`](./CTC_TRAINING.md) | Training loop details; the uncapped notebook follows the same loop structure |
| [`CTC_FILE_REFERENCE.md`](./CTC_FILE_REFERENCE.md) | Per-file API reference for `src/ctc/`; for `src/CRNN_CTC_Uncapped/` see §4 above |
