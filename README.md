# Digit Sequence Reader

A deep learning project that reads a **variable-length sequence of handwritten digits** from a single stitched image and outputs the correct string of numbers.

The repository features two distinct approaches to sequence reading:
1. **Autoregressive Sequence-to-Sequence (Seq2Seq)**: Uses a `CNN + Bidirectional LSTM` encoder, `Bahdanau Attention`, and an `LSTM Decoder` loop to output characters step-by-step.
2. **Parallel Connectionist Temporal Classification (CTC)**: Uses a `2D CNN` encoder, a height mean collapse layer, a `1D Dilated Residual CNN` encoder, a linear classifier head, and a `CTC Loss` function. This model is non-autoregressive and excels at **generalizing to sequence lengths far beyond the training distribution** (length extrapolation).

---

## 📖 Key Documentation

* **[Comprehensive System Documentation (PROJECT_DOCUMENTATION.md)](PROJECT_DOCUMENTATION.md)**: The main, detailed guide covering the architecture of both models, mathematical formulations, the data pipeline (EMNIST + QMNIST + USPS), curriculum learning, the CTC width guarantee, and step-by-step guides for training and inference.
* **[CTC Architecture deep dive](docs/CTC_ARCHITECTURE.md)**: Technical trace of shapes and layers in the CTC model.
* **[Length Extrapolation detailed analysis](docs/CTC_EXTRAPOLATION.md)**: Explanation of why Seq2Seq fails at extrapolation and why CTC handles it gracefully.
* **[CTC Training guidelines](docs/CTC_TRAINING.md)**: Curriculum settings, free-logits optimization, and troubleshooting.
* **[CTC Inference guidelines](docs/CTC_INFERENCE.md)**: Greedy decoding details, Levenshtein edit distance, and visualization tools.
* **[File reference guide](docs/CTC_FILE_REFERENCE.md)**: Quick pointers to public APIs for all scripts.

---

## 🚀 Quick Start (CTC Model)

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Images
To synthesize 20 sample images from the training distribution:
```bash
make ctc-generate
```

### 3. Training
To train the CTC model locally (automatically updating the augmentation and length curriculum):
```bash
make ctc-train
```
Checkpoints will be saved to `./model/checkpoints_ctc/best_ctc.pt`.

### 4. Inference and Visualization
To run inference on a test image (e.g., `samples/test.png`) and view the temporal prediction plot:
```bash
make ctc-infer
```

You can specify a different image and supply ground truth to compute Character Error Rate (CER):
```bash
make ctc-infer IMAGE=samples/sample_L7_8675476.png
```

### 5. Length Extrapolation Evaluation
To benchmark the CTC model on sequence lengths $1 \to 50$ (trained on lengths $\le 12$):
```bash
make ctc-eval-extrap
```
This plots Sequence Accuracy and CER against length, saving the result to `./model/metrics/ctc_length_extrapolation.png`.

---

## 🚀 Quick Start (Seq2Seq Model)

### 1. Training
To train the Seq2Seq model:
```bash
python -m src.seq2seq.train --drive_path ./model
```

### 2. Inference
To predict an image and display the soft attention spotlight heatmap:
```bash
make infer
```

---

## 🏛️ Architecture & Data At A Glance

### Parallel CTC Flow
```
Input Image [B, 1, 64, W] ──► 2D CNN ──► Collapsed Sequence [B, 512, W//8] ──► Dilated 1D CNN ──► Linear Head ──► CTC Greedy Decoder
```

### Dynamic Data Generation
Training sequences are generated **on-the-fly** from a combined pool of **~307,000 real handwritten digits** (EMNIST, QMNIST, USPS). The dataset loader dynamically concatenates characters with variable gaps and curriculum-based overlaps. 

Albumentations augmentations (brightness, noise, blur, dropout) scale in intensity over the first 10 epochs. The generator strictly enforces the **CTC width guarantee** ($W \ge \max(64, L \times 16)$) to ensure the network downsampling doesn't violate the CTC sequence length condition ($T \ge L$).

---

## 📊 Example Predictions

### Input Image:
![Sample Image](img-readme/sample_5_8516313.png)

### Seq2Seq Predicted Sequence & Soft Attention:
![Attention Heatmap](img-readme/8516313.png)
*(The attention spotlight sweeps naturally from left to right as the decoder step increases.)*

### CTC Predicted Sequence & Temporal Argmax Trace:
When running `make ctc-infer`, the system plots a frame-by-frame prediction trace (with intermediate `BLANK` peaks) indicating the alignment discovered by CTC.

---

## 💻 Google Colab Notebooks
You can also run training on a Google Colab GPU:
* **[Seq2Seq Notebook](train_colab.ipynb)**
* **[CTC Notebook](train_colab_ctc.ipynb)**
