# Training the CTC Model

This document covers everything between `make ctc-train` and the moment
`best_ctc.pt` lands in `./checkpoints_ctc/`. It's the "how" companion
to [`CTC_ARCHITECTURE.md`](./CTC_ARCHITECTURE.md).

---

## 1. The training command

```bash
make ctc-train
```

Which expands to:

```bash
python -m src.ctc.train --drive_path ./model --epochs 30 --batch_size 64 --lr 1e-3
```

Override anything you want from the command line:

```bash
python -m src.ctc.train --drive_path /content/drive/MyDrive/digit-sequence-reader \
    --epochs 50 --batch_size 32 --lr 5e-4 --resume ./model/checkpoints_ctc/best_ctc.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--drive_path` | *(required)* | Where to save checkpoints/metrics |
| `--epochs`     | 30 | Total epochs (curriculum max kicks in at 11) |
| `--batch_size` | 64 | Items per batch |
| `--lr`         | 1e-3 | Initial learning rate |
| `--resume`     | `""` | Path to a previous `best_ctc.pt` to continue from |

**Output structure:**
```
{drive_path}/
├── checkpoints_ctc/
│   └── best_ctc.pt            # best validation seq_acc so far
└── metrics/
    └── ctc_metrics.csv        # per-epoch metrics
```

---

## 2. What the training loop does

Each epoch:

1. **Build a fresh `InfiniteCTCDataset`** for the train split, with the
   augmentation intensity scaled by the current epoch. (The dataset is
   an `IterableDataset` so it streams fresh samples every batch.)
2. **Train for `train_size // batch_size` steps.** `train_size=100,000`
   by default, so 1,562 steps per epoch at batch size 64.
3. **Validate** on the val loader (10,000 clean samples).
4. **Save `best_ctc.pt`** if `val_seq_acc` improved.
5. **Update the LR scheduler** (`ReduceLROnPlateau` on val loss).
6. **Check early stopping** (12 epochs without improvement → stop).

After all epochs (or early stop), **run a final test-set evaluation**
on 10,000 held-out clean samples.

---

## 3. The CTC loss

```python
self.ctc_loss = nn.CTCLoss(blank=10, zero_infinity=True)
```

### What it computes

For a batch:
- `log_probs` `[T, B, 11]` — log-softmaxed frame-level predictions
- `targets` `[sum(L_i)]` — concatenated raw digit lists
- `input_lengths` `[B]` — all equal to `T = max_w // 8` (we use
  the padded batch-max, with the right padding interpreted as
  BLANK by CTC)
- `target_lengths` `[B]` — actual number of digits per item

CTC marginalises over all valid frame-level alignments and returns
the negative log-likelihood. Minimising it = maximising the
probability of *any* valid frame sequence that, after collapsing
duplicates and removing BLANKs, gives the target.

### What `zero_infinity=True` does

If `T < L` for any item (no valid alignment), CTC returns `+Inf`.
Setting this flag replaces `+Inf` (and any `NaN` gradient) with 0,
so training doesn't crash on a single bad batch. We **prevent** this
from happening by enforcing `T ≥ 2L` in the dataset — see
[`§ 6. The width guarantee`](#6-the-width-guarantee-ctc-safety-net).

---

## 4. The "free logits" optimisation

The model returns `(loss, logits)` on the training forward pass:

```python
def forward(self, images, targets=None, target_lengths=None):
    ...
    if targets is not None:
        loss = self.ctc_loss(...)
        return loss, logits     # <-- the trick
    return logits
```

The training loop reuses these logits for metric logging
(`greedy_decode(logits.detach())`) instead of doing a second forward
pass. This **saves ~30% of compute on log steps** (every 50 steps by
default) — over 30 epochs that's hours saved on a single GPU run.

---

## 5. The curriculum

`max_seq_len` grows linearly from 7 to 12 over epochs 11-30:

```python
def get_curriculum_max_len(epoch, config):
    base   = config['max_seq_len']         # 7
    final  = config.get('max_seq_len_final', 12)
    warmup = config.get('aug_warmup_epochs', 10)
    if epoch <= warmup:
        return base
    progress = min(1.0, (epoch - warmup) / 20.0)
    return int(base + progress * (final - base))
```

The intuition: train on short sequences first so the model learns
the "one digit = one chunk of frames" mapping cleanly, then gradually
introduce longer sequences. By the time `max_seq_len=12`, the model
has solid per-digit representations and can chain them together.

The validation and test loaders always use `max_seq_len=7` (no
curriculum tail) so val/test metrics are consistent across epochs.

---

## 6. The width guarantee (CTC safety net)

`PyTorch CTCLoss` will return `+Inf` if `T < L` for any item. To
prevent this:

```python
# In make_sequence():
min_w = max(config['min_image_width'], L * config['width_per_digit'])
target_w = min_w + random.randint(0, max(0, min_w // 2))
target_w = ((target_w + 7) // 8) * 8    # snap to multiple of 8
```

With `width_per_digit = 16`:
- `L = 7` → `min_w = 112` → `T = 112 // 8 = 14` (≥ 2 × 7 = 14 ✓)
- `L = 12` → `min_w = 192` → `T = 24` (≥ 2 × 12 = 24 ✓)
- `L = 50` → `min_w = 800` → `T = 100` (≥ 100 ✓)

This means `T ≥ 2L` for every training sample, so CTC always has at
least one valid alignment and the loss is always finite.

The `collate_fn` then uses `max_w // 8` as the input length for
**every** item in the batch (since the right padding contributes
empty features that CTC will label as BLANK).

---

## 7. Augmentation pipeline

`get_digit_aug_pipeline(augment, config, epoch)` returns a callable
that takes a PIL image of one digit and returns a `1 × 64 × 64`
tensor. Components, in order:

| Transform | When | Intensity |
|-----------|------|-----------|
| `A.Affine(rotate, shear)` | only if `seq_rotation > 0` | scaled by `epoch / 10` |
| `A.RandomBrightnessContrast` | always | `p=0.5`, scaled by intensity |
| `A.GaussianNoise` or `A.MotionBlur` | always | one-of, `p=0.4 × intensity` |
| `A.CoarseDropout` | always | `p=0.3 × intensity` |
| `A.Resize(64, 64)` | always | n/a |
| `A.Normalize(mean=0, std=1)` | always | n/a |

`intensity = min(1.0, epoch / aug_warmup_epochs)`. So in early
epochs the model sees mostly-clean digits; in later epochs it sees
the full noisy / rotated / contrast-shifted / dropout-pocked
version. The default `seq_rotation=0` keeps digits upright
(readable); set to 3-5 if you want to add a small tilt.

---

## 8. Optimisation & scheduling

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| Optimiser | `AdamW(lr=1e-3, weight_decay=1e-4)` | AdamW > Adam for this size of model |
| Scheduler | `ReduceLROnPlateau(patience=5, factor=0.5, min=1e-6)` | Drops LR when val loss plateaus |
| Gradient clip | `clip_grad_norm_(..., 1.0)` | Prevents the dilated 1D stack from blowing up |
| Early stop | 12 epochs without improvement | Stops around epoch 25-30 in practice |

---

## 9. Metrics

For every batch we compute:

- **Sequence accuracy** = exact match of decoded digits vs ground truth.
- **CER (Character Error Rate)** = `Levenshtein(pred, gt) / |gt|`.

Reported per epoch (averaged across the last `log_every=50` batches of
training, and across the entire validation set).

CSV format (`metrics/ctc_metrics.csv`):

```
epoch, train_loss, val_loss, train_seq_acc, val_seq_acc, train_char_acc, val_char_acc, lr
1,    2.41,       2.32,     0.102,         0.121,       0.41,           0.43,         1.0e-3
...
```

---

## 10. Reproducing a run

```bash
# Same seed, same results
python -m src.ctc.train --drive_path ./run1 --epochs 30
# (set torch.manual_seed(42) etc. inside train.py if you want a fixed seed —
#  it's currently left random for genuine multi-run averaging)
```

The training is deterministic **up to the GPU/CUDA** — the dilated
1D CNN has no stochastic components (no dropout in the encoder
proper, just `Dropout(p=0.1)` between the two convs in each block),
so multi-seed runs are highly reproducible.

---

## 11. Common training issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss stays at `~ln(11) ≈ 2.4` for the first 1000 steps | Bug in the data pipeline (always returning same image) | Visualise a few `images` tensors — they should be different |
| Loss goes to `NaN` | `T < L` somewhere | Check the width guarantee; enable `zero_infinity=True` (already on) |
| Val seq-acc plateaus at < 50% | Curriculum max too high for the LR schedule | Lower `--lr` to `5e-4` and increase `--epochs` to 50 |
| Training speed < 5 it/s on a T4 | Too many `num_workers` for the small batch | Set `num_workers=2` (default) |
| `RuntimeError: out of memory` | Batch too large for the GPU | Lower `--batch_size` to 32 or 16 |

For end-to-end debugging, see [`CTC_FILE_REFERENCE.md`](./CTC_FILE_REFERENCE.md).
