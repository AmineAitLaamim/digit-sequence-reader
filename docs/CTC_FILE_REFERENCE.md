# File-by-File Reference

This document lists every Python module in `src/ctc/`, its public
API, and what each function/class is for. Use it as a quick lookup
when navigating the code.

---

## 📁 `src/ctc/config.py` — hyperparameters

A single dictionary literal. **No executable code** apart from the
dict definition. Keys are grouped into comments.

```python
from .config import config
config['vocab_size']        # 11
config['BLANK_IDX']         # 10
config['hidden_dim']        # 256
config['dilations']         # [1, 2, 4, 8]
config['width_per_digit']   # 16
config['lr']                # 1e-3
config['epochs']            # 30
...
```

The values can be overridden by the CLI flags in `train.py`,
`inference.py`, etc.

---

## 📁 `src/ctc/model.py` — the architecture

### Classes

| Name | Purpose |
|------|---------|
| `_ConvBlock2D(in, out, k, pool)` | Single Conv-BN-GELU-MaxPool block (internal) |
| `CNN2DEncoder()` | 4-block 2D CNN: `[B,1,64,W] → [B,512,8,W//8]` |
| `ResidualConv1DBlock(in, out, k, dil, drop)` | 1D dilated residual block |
| `DilatedCNN1DEncoder()` | 4-block dilated 1D CNN: `[B,512,T] → [B,256,T]` |
| `CRNN_CTC()` | The full model. Returns `(loss, logits)` when targets given, else `logits` only. |

### Functions

| Name | Purpose |
|------|---------|
| `greedy_decode(logits)` | Collapse consecutive duplicates, remove BLANKs. Returns `list[list[int]]`. |

### `CRNN_CTC` public methods

```python
model = CRNN_CTC()

# Inference
logits = model(images)            # [B, T, 11]

# Training
loss, logits = model(images, targets=targets, target_lengths=target_lengths)

# Save / load
torch.save({'model_state_dict': model.state_dict(), ...}, 'ckpt.pt')
model.load_state_dict(torch.load('ckpt.pt')['model_state_dict'], strict=False)
```

### `__main__` block

A shape-verification harness. Run it with `python -m src.ctc.model`
to assert that:
- `model(dummy_img, targets=..., target_lengths=...)` returns a tuple `(loss, logits)`
- `loss.item()` is finite
- `logits.shape == (2, 20, 11)` for `B=2, W=160`

This is the same check the Colab notebook runs in cell 4.

---

## 📁 `src/ctc/dataset.py` — data pipeline

### Functions

| Name | Purpose |
|------|---------|
| `build_multidigit_bank(data_path)` | Load EMNIST Digits + QMNIST + USPS into `{0..9: list[Image]}` |
| `get_digit_aug_pipeline(augment, config, epoch)` | Returns a callable `pil → tensor` (augmented) or just `Resize + ToTensor` (clean) |
| `get_curriculum_max_len(epoch, config)` | `7 → 12` over epochs 11-30 |
| `make_sequence(bank, aug, config, augment, epoch)` | Generate one `(image, digits_list)` pair; **enforces width guarantee** |

### Classes

| Name | Purpose |
|------|---------|
| `InfiniteCTCDataset(bank, config, size, augment, epoch)` | `IterableDataset` that streams `make_sequence` outputs |

### Functions (continued)

| Name | Purpose |
|------|---------|
| `collate_fn(batch)` | Pads images + flattens targets + computes `input_lengths` and `target_lengths`. Returns the **4-key dict** that `nn.CTCLoss` needs. |
| `get_dataloaders(data_path)` | Returns `(bank, val_loader, test_loader)`. Train loader is rebuilt each epoch in `train.py`. |

### The 4-key dict contract

```python
batch = next(iter(val_loader))
assert set(batch.keys()) == {
    'images',         # [B, 1, 64, max_w]
    'targets',        # [sum(L_i)]           1D flattened, no SOS/EOS
    'input_lengths',  # [B]                   all == max_w // 8
    'target_lengths', # [B]                   actual digit count per item
}
```

The contract is **asserted** in the `__main__` block of `dataset.py`.

---

## 📁 `src/ctc/train.py` — training loop

### Public functions

| Name | Purpose |
|------|---------|
| `compute_metrics(decoded, target_lengths, targets_flat)` | Returns `(seq_acc, cer)` |
| `train_one_epoch(model, loader, optim, device, steps, log_every=50)` | Train loop with **free logits** optimisation |
| `validate(model, loader, device)` | Validation loop; also returns `(loss, seq_acc, char_acc)` |
| `save_checkpoint(model, optim, epoch, val_loss, val_seq_acc, path)` | Standard checkpoint dict |
| `main()` | CLI entry point (`python -m src.ctc.train`) |
| `_levenshtein(a, b)` | (Internal) Standard DP for CER computation |

### CLI flags

```
--drive_path   (required)   Where to save checkpoints/metrics
--epochs       (default 30) Total epochs
--batch_size   (default 64) Items per batch
--lr           (default 1e-3) Initial learning rate
--resume       (default "")  Path to best_ctc.pt to continue
```

### Checkpoint format

```python
{
    'epoch':                int,
    'model_state_dict':         state_dict,
    'optimizer_state_dict':     state_dict,
    'val_loss':             float,
    'val_seq_acc':          float,
    'config':               config dict,
}
```

### Key implementation notes

- The training loop **does not** use `batch['input_lengths']` because
  the model computes it internally from `feat.size(3)` (the actual CNN
  output width). The key is still in the dict (so external code can
  inspect the batch), but it's not needed for the loss.
- The training loop uses the **free logits** trick: the model returns
  `(loss, logits)` and the loop reuses `logits.detach()` for metric
  logging instead of doing a second forward pass.
- Early stopping is `12` epochs without `val_seq_acc` improvement.

---

## 📁 `src/ctc/inference.py` — single-image inference

### Functions

| Name | Purpose |
|------|---------|
| `preprocess_image(path)` | Load image, resize to `H=64`, snap width to multiple of 8. Returns `(tensor, PIL_image)`. |
| `_levenshtein(a, b)` | Same as in `train.py` (used for CER on the inference side) |
| `auto_extract_gt(image_path)` | Pull digits from the end of the filename via `re.search(r'(\d+)$', basename)` |
| `predict(image_path, ckpt, visualize, ground_truth)` | Full inference. Returns `(pred_string, metrics_or_None)` |
| `main()` | CLI entry point (`python -m src.ctc.inference`) |

### CLI flags

```
--image         (required)  Path to the test image
--checkpoint    (required)  Path to best_ctc.pt
--visualize                 Show 2-panel plot (image + per-frame argmax)
--ground-truth  (default None)  Explicit GT digit string
--no-gt                     Skip GT comparison even if filename has digits
```

### The predict() return value

```python
pred_string, metrics = predict("img.png", "ckpt.pt", visualize=True, ground_truth="12345")

# pred_string: "12345"  (str)
# metrics:    dict or None
#   {
#       'ground_truth': '12345',
#       'pred':         '12345',
#       'seq_match':    True,
#       'cer':          0.0,
#       'edit_distance': 0,
#       'gt_length':    5,
#       'pred_length':  5,
#   }
```

---

## 📁 `src/ctc/generate_samples.py` — 20 random samples

### Usage

```bash
make ctc-generate
# or
python -m src.ctc.generate_samples
```

Generates 20 PNGs of CTC-style sequences (filename
`ctc_sample_<i>_<digits>.png`) in `./samples/`, using the curriculum
epoch of the config.

### Public functions

| Name | Purpose |
|------|---------|
| `main()` | The entire script (no public functions) |

Internally uses `build_multidigit_bank`, `get_digit_aug_pipeline`,
`make_sequence` from `dataset.py`.

---

## 📁 `src/ctc/generate_one.py` — ONE sample of a specific length

### Usage

```bash
# Default filename: samples/sample_L7_<digits>.png
make ctc-gen-one L=7

# Custom output path
make ctc-gen-one L=25 OUT=my_test.png

# Multiple samples (each with a different random digit string)
make ctc-gen-one L=12 COUNT=5

# With training-style augmentation
make ctc-gen-one L=7 AUG=1

# Reproducible
make ctc-gen-one L=7 SEED=42
```

### Functions

| Name | Purpose |
|------|---------|
| `generate_one(length, augment, out_path, epoch, seed)` | Generate and save a single sample. Returns `(out_path, digit_str)`. |
| `_get_bank_and_aug(augment, epoch)` | Internal: caches the digit bank + augmentation pipeline across calls |
| `main()` | CLI entry point |

### Module-level cache

The digit bank is **built once** (downloading EMNIST/QMNIST/USPS takes
~30s) and cached at module level. Calling `generate_one` N times only
re-builds it on the first call.

---

## 📁 `src/ctc/evaluate_extrapolation.py` — length benchmark

### Usage

```bash
make ctc-eval-extrap
# or with custom lengths
python -m src.ctc.evaluate_extrapolation \
    --checkpoint ./model/best_ctc.pt \
    --lengths 5,10,15,20,30,50,75,100 \
    --n_samples 500 \
    --out_dir ./my_metrics
```

### CLI flags

```
--checkpoint  (required)  Path to best_ctc.pt
--out_dir     (default ./metrics)  Where to save the plot + JSON
--lengths     (default 1,3,5,7,9,12,15,20,25,30,40,50)  Comma-separated
--n_samples   (default 200)  Samples per length
```

### Functions

| Name | Purpose |
|------|---------|
| `evaluate_at_length(model, bank, aug, L, n_samples, device)` | Returns `(seq_acc, cer, examples)` for one length |
| `main()` | CLI entry point. Saves `<out_dir>/ctc_length_extrapolation.{png,json}` |

### Output format (JSON)

```json
{
  "checkpoint":    "best_ctc.pt",
  "n_samples":     200,
  "train_max_len": 12,
  "results": {
    "1":  {"seq_acc": 1.0,  "cer": 0.0},
    "3":  {"seq_acc": 0.99, "cer": 0.01},
    "12": {"seq_acc": 0.97, "cer": 0.03},
    "20": {"seq_acc": 0.85, "cer": 0.12},
    "50": {"seq_acc": 0.42, "cer": 0.31}
  }
}
```

---

## 📁 `train_colab_ctc.ipynb` — Google Colab notebook

The parallel of the original `train_colab.ipynb`, but wired to the
CTC entry points. **19 cells** (9 markdown + 10 code):

| # | Type | Purpose |
|--:|------|---------|
| 1 | markdown | Title + architecture blurb |
| 2 | markdown | §1 Mount Drive |
| 3 | code | Mount Google Drive |
| 4 | markdown | §2 Clone repo + install |
| 5 | code | `git clone` + `pip install` |
| 6 | markdown | §3 Verify GPU |
| 7 | code | `torch.cuda.is_available()` check |
| 8 | markdown | §4 Shape sanity check |
| 9 | code | `!python -m src.ctc.model` (asserts `[2, 160] → [2, 20, 11]`) |
| 10 | markdown | §5 Train |
| 11 | code | `!python -m src.ctc.train --drive_path ...` |
| 12 | markdown | §6 Inspect training curves |
| 13 | code | `pandas` + `matplotlib` to plot loss/seq-acc/char-acc |
| 14 | markdown | §7 Generate sample images |
| 15 | code | `!python -m src.ctc.generate_samples` |
| 16 | markdown | §8 Length-extrapolation check |
| 17 | code | Loop over `L ∈ {3,7,12,20,30}` and print true vs pred |
| 18 | markdown | §9 Inference on custom image |
| 19 | code | `files.upload()` + `!python -m src.ctc.inference` |

All cell IDs (for the CTC module) point to `model/best_ctc.pt` or
`checkpoints_ctc/best_ctc.pt` consistently.

---

## 📁 `Makefile` — every CTC target

```makefile
# Inference (with optional accuracy via GT/IMAGE/NO_GT args)
ctc-infer

# Generate 20 random samples
ctc-generate

# Generate ONE sample of an exact length
ctc-gen-one  L=<int>  [OUT=path]  [COUNT=N]  [AUG=1]  [SEED=N]

# Train (saves to ./checkpoints_ctc/best_ctc.pt)
ctc-train

# Length extrapolation benchmark
ctc-eval-extrap
```

Run `make` (no target) to list them, or check the file directly.

---

## 📁 File sizes (lines of code)

```
src/ctc/
  config.py                      ~85   (a single dict literal + comments)
  model.py                      ~325   (the architecture + shape tests)
  dataset.py                    ~250   (data pipeline + collate)
  train.py                      ~270   (training loop with free logits)
  inference.py                  ~155   (inference + GT metrics)
  generate_samples.py            ~50   (20-sample generator)
  generate_one.py                ~165  (single-sample generator with bank cache)
  evaluate_extrapolation.py     ~190  (length benchmark + plot)
```

Total: ~1,500 lines of focused, well-commented Python.
