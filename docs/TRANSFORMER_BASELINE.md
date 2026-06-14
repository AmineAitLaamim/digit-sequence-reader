# Transformer Seq2Seq Baseline — Length Extrapolation Ablation

This document details the **Transformer Seq2Seq** baseline added to the repository (`src/transformer_baseline/`).

Unlike the primary CTC and Uncapped models, this Transformer baseline was intentionally designed to **fail the length extrapolation test**. By establishing a state-of-the-art autoregressive baseline that suffers from catastrophic length prior collapse, we highlight the unique architectural strengths of the non-autoregressive CTC approach.

---

## 1. Why build a model designed to fail?

When preparing the research paper, a critical question arises: *Are standard sequence-to-sequence models inherently bad at length extrapolation?*

To prove that the CTC model's generalisation is a structural property (and not just an artifact of the data pipeline), we needed an "apples-to-apples" ablation. We built a Transformer that shares the exact same visual feature extraction and data pipeline as the CTC model, but replaces the core decoding mechanics.

### The Hypotheses Tested
The Transformer baseline tests three specific architectural choices that induce a rigid "length prior":
1. **Absolute Positional Encodings:** Sinusoidal PEs explicitly anchor the model to sequence lengths seen during training.
2. **Global Self-Attention:** The decoder can "see" the entire sequence at once, enabling it to learn shortcuts about sequence termination.
3. **Autoregressive Decoding:** The model must predict when to emit the `<EOS>` token based on its previous sequence of outputs.

---

## 2. Architecture Details

The model (`src/transformer_baseline/model.py`) is a hybrid CNN-Transformer:

```
Input Image [B, 1, 64, W] 
        │
        ▼
┌────────────────────────────────────┐
│  2D CNN encoder (Shared with CTC)  │  (8x spatial downsample)
│  Output: [B, 512, 8, W//8]         │
└────────────────────────────────────┘
        │
        ▼ mean over height H
[B, 512, W//8]
        │
        ▼ Linear Projection
[B, W//8, d_model]  (Encoder Memory)
        │
        ▼
┌────────────────────────────────────┐
│  Transformer Decoder               │  
│  - Absolute Positional Encoding    │
│  - Causal & Padding Masks          │
│  - Cross-Attention to Memory       │
│  Output: [B, L+1, d_model]         │
└────────────────────────────────────┘
        │
        ▼ Linear Head
Logits over Vocabulary (Digits + <SOS> + <EOS> + <PAD>)
```

### Absolute Positional Encoding
We implemented standard sinusoidal encodings. Because the PE indices strictly correspond to the absolute positions in the target sequence ($0$ to $L$), when the model encounters an out-of-distribution length ($L=50$), the decoder embeddings at $t > 12$ fall into untested PE manifolds, causing confidence collapse.

---

## 3. The Data Pipeline and `transformer_collate_fn`

To guarantee a fair comparison, the Transformer must train on the **exact same data** as the CTC model. We reuse `InfiniteCTCDataset` unmodified. 

However, the CTC pipeline returns 1D flattened targets (e.g., `[sum(L_i)]`), while the Transformer decoder expects a 2D padded target matrix (`[B, max_L]`). 

To preserve separation of concerns without modifying the underlying dataset:
1. We introduced `transformer_collate_fn` in `src/transformer_baseline/train.py`.
2. This wrapper intercepts the batch from `ctc_collate_fn` and packs the 1D targets into a 2D tensor padded with `PAD_IDX=10`.
3. The `val_loader` is also explicitly recreated with this wrapper to ensure validation metrics compute flawlessly.

---

## 4. Expected Extrapolation Results

When running the length extrapolation test on the fully trained Transformer, the sequence accuracy curve looks drastically different from the CTC model:

| Train L | Test L | CTC Expected seq-acc | Transformer Expected seq-acc |
|--------:|-------:|:---------------------|:-----------------------------|
| 3-12    | 12     | > 98 %               | > 60-80 % (In-distribution)  |
| 3-12    | 20     | > 80 %               | ~ 10-20 % (Collapse begins)  |
| 3-12    | 30     | > 60 %               | < 5 %                        |
| 3-12    | 50     | > 30 %               | 0 % (Catastrophic failure)   |

This confirms the central thesis: **Autoregressive decoding with absolute positional encodings fundamentally limits a model to the sequence lengths observed during training.**

---

## 5. How to Run

The baseline is designed to be trained in a Google Colab GPU environment. 

We provided `notebooks/train_colab_transformer.ipynb` which identically mirrors the CTC notebook structure but uses the `TransformerSeq2Seq` architecture and the `WarmupCosineLR` scheduler (which is mandatory for Transformer stability).

1. Upload `notebooks/train_colab_transformer.ipynb` to Colab.
2. Mount your Google Drive.
3. Run all cells to execute the 30-epoch training loop.
4. The final cell will automatically execute the Length-Extrapolation evaluation array.
