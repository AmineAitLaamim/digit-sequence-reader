# CRNN + CTC — Digit Sequence Reader

A non-autoregressive, parallel-decoding model that reads variable-length
sequences of handwritten digits from a single grayscale image. Built
specifically to **generalise to sequence lengths far beyond the training
distribution** without learning a length prior.

---

## 📚 Documentation Index

| File | What you'll find |
|------|------------------|
| **[`CTC_ARCHITECTURE.md`](./CTC_ARCHITECTURE.md)** | Why we chose CRNN + CTC, dimensional walkthrough of every layer, why each design choice (2D CNN, dilated 1D, no Transformer) prevents shortcut learning. **Start here.** |
| **[`CTC_TRAINING.md`](./CTC_TRAINING.md)** | How training works, the CTC loss, the curriculum, the width guarantee that prevents `Inf` losses, the data pipeline, and the optimisation tweaks. |
| **[`CTC_INFERENCE.md`](./CTC_INFERENCE.md)** | How to run the trained model on a single image, how the greedy decoder works, how to read the FINAL RESULT banner, and the full Makefile command reference. |
| **[`CTC_EXTRAPOLATION.md`](./CTC_EXTRAPOLATION.md)** | The key selling point of the model: how and why it generalises to unseen lengths. Includes the `evaluate_extrapolation.py` script and how to interpret the length-vs-accuracy plot. |
| **[`CTC_FILE_REFERENCE.md`](./CTC_FILE_REFERENCE.md)** | File-by-file reference for every module in `src/ctc/`, with the public API of each. |
| **[`ABLATION_UNCAPPED.md`](./ABLATION_UNCAPPED.md)** | The `CRNN_CTC_Uncapped` ablation study — the one architectural change (6 dilated blocks, RF≈252), the hypothesis it tests, expected results, and how to reproduce the experiment. |

---

## 🚀 Quickstart

### Local
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train on a GPU box
make ctc-train           # saves to ./checkpoints_ctc/best_ctc.pt

# 3. Inference on a test image (with optional accuracy metrics)
make ctc-infer IMAGE=samples/sample_L7_1234567.png   # auto-extracts GT
make ctc-infer GT=12345                               # explicit GT

# 4. The headline experiment: length extrapolation
make ctc-eval-extrap   # synthesises L in {1,3,5,...,50} and plots seq-acc + CER
```

### Google Colab

| Notebook | Purpose |
|----------|---------|
| [`train_colab_ctc.ipynb`](../train_colab_ctc.ipynb) | Train the **baseline** `CRNN_CTC` model end-to-end. |
| [`notebooks/train_colab_ctc_uncapped.ipynb`](../notebooks/train_colab_ctc_uncapped.ipynb) | Train the **ablation** `CRNN_CTC_Uncapped` model and run the length-extrapolation comparison. |

---

## 🏛️ Architecture at a Glance

```
Input image [B, 1, 64, W]              (variable width W)
        │
        ▼
┌────────────────────────────────────┐
│  2D CNN encoder                    │   H: 64 → 8      (8x downsample)
│  4 blocks, channels [64,128,256,512]   W: W → W//8    (8x downsample)
│  Output: [B, 512, 8, W//8]          │
└────────────────────────────────────┘
        │
        ▼  mean over height H
[B, 512, W//8]
        │
        ▼
┌────────────────────────────────────┐
│  1D Dilated Residual CNN           │   receptive field ≈ 60 steps
│  4 blocks, dilations 1, 2, 4, 8    │   (but never sees the full sequence)
│  Output: [B, 256, W//8]             │
└────────────────────────────────────┘
        │
        ▼
Linear classifier → [B, W//8, 11]   (10 digits + 1 BLANK)
        │
        ▼
Greedy CTC decoder → digit string
```

### Why this design works

1. **2D CNN = no length prior.** A convolutional filter has a fixed
   spatial kernel; it never sees the *whole* image. So the model
   physically cannot learn "wider image → longer sequence".
2. **1D Dilated CNN = no global context.** With dilations {1,2,4,8} and
   kernel size 3, the receptive field is ≈ 60 time steps. Plenty of
   local context to resolve a digit and its neighbours, but never
   enough to see the entire sequence. So the model can never condition
   its prediction on "I have seen all the digits already".
3. **CTC = no exposure bias.** Every output frame is conditioned only
   on the corresponding input frame, not on previous outputs. So errors
   don't compound the way they do in autoregressive decoders.
4. **Width guarantee = no Inf loss.** The dataset synthesises images
   with `W ≥ max(64, L × 16)`, so after the 8x downsample `T = W//8 ≥
   2L`. This guarantees CTC can always find a valid alignment.

---

## 📊 Headline Result

| Train L | Test L | Expected seq-acc | Why |
|--------:|-------:|:-----------------|-----|
| 3-7     | 7      | > 95 %           | In-distribution |
| 3-12    | 12     | > 95 %           | Curriculum max  |
| 3-12    | 20     | > 80 %           | 1.7× extrapolation |
| 3-12    | 30     | > 60 %           | 2.5× extrapolation |
| 3-12    | 50     | > 30 %           | 4.2× extrapolation |

*(Numbers are illustrative — the autoregressive Seq2Seq model typically
collapses past 1.2× extrapolation.)*

Run `make ctc-eval-extrap` after training to generate the actual curve.

---

## 🗂️ Repo Layout (CTC-specific files only)

```
src/
├── ctc/                          # Baseline (production) model
│   ├── config.py                 # all hyperparameters, vocab, paths
│   ├── model.py                  # CRNN_CTC architecture + greedy_decode
│   ├── dataset.py                # data pipeline + collate_fn (4-key dict)
│   ├── train.py                  # training loop with CTC loss + free logits
│   ├── inference.py              # single-image inference + accuracy metrics
│   ├── generate_samples.py       # 20 random samples (curriculum epoch)
│   ├── generate_one.py           # ONE sample of an exact length
│   └── evaluate_extrapolation.py # length-vs-accuracy benchmarking
└── CRNN_CTC_Uncapped/            # Ablation model (uncapped receptive field)
    ├── __init__.py
    └── model.py                  # CRNN_CTC_Uncapped, RF≈252, dilations [1,2,4,8,16,32]
docs/                             # this folder
train_colab_ctc.ipynb             # Colab notebook — baseline model
notebooks/
└── train_colab_ctc_uncapped.ipynb  # Colab notebook — ablation model
```

---

## 📚 Further reading

- Graves et al., *Connectionist Temporal Classification* (2006) — the original CTC paper.
- Hannun et al., *Deep Speech* (2014) — a classic CTC end-to-end speech model.
- This repo: see [`CTC_EXTRAPOLATION.md`](./CTC_EXTRAPOLATION.md) for the theoretical argument that dilated CNN + CTC *cannot* learn a length prior.
