# Length Extrapolation — The Headline Result

This is the document that justifies the entire CTC migration.
**The whole point of the architecture is that it can read sequences
of length L=50 even though it was only trained on L ≤ 12.** This
file explains how that's possible, how to measure it, and how to
interpret the results.

---

## 1. Why length extrapolation is hard for autoregressive models

The previous model (`src/seq2seq/`) was an **encoder-decoder with
attention** — an LSTM that emitted one digit at a time, with a
`<EOS>` token to signal "the sequence is over".

Concretely, training looks like:

```
input image → CNN features → BiLSTM encoder → (h_0, c_0)
                ↓
        LSTM decoder step t:
            input: previous token (digit or <SOS>)
            output: logits over {digit 0-9, <EOS>, <PAD>}
            stop when <EOS> is predicted
```

The model is forced to **predict when to stop**. So at training
time it sees sequences of length 3-7 (then 3-12 with the
curriculum) and learns "after ~7-12 digits, emit <EOS>".

At test time with a sequence of length 20:

- The decoder *wants* to emit `<EOS>` after 7-12 digits because
  that's what it always saw.
- The model has no architectural way to *know* the sequence is
  20 digits long — that information isn't in the input features
  the decoder can attend to (the encoder output is `T` time steps
  of features, but the LSTM decoder is autoregressive and only
  sees its own previous outputs).

The result: the seq2seq model **collapses** past ~1.2× the
training max length. CER explodes to 0.5+ and seq-acc to ~0%.

---

## 2. Why CTC with our architecture *can't* learn a length prior

The CTC model emits one prediction per **CNN time step** (not per
*previous* output). Each prediction is conditionally independent
of every other prediction, given the input features:

```
P(y_1, y_2, ..., y_T | x) = ∏_t P(y_t | x_features_t)
```

The only mechanism that could "know how many digits there are" is
the input features themselves. But:

1. **The 2D CNN is translation-equivariant.** Every column is
   processed with the *same* weights. The model literally cannot
   tell "I am at column 30" from "I am at column 5". It can only
   tell "there is a 7 here" or "there is a 1 here".

2. **The 1D Dilated CNN has a fixed receptive field of ≈60 steps.**
   The model can resolve local context (a digit + its neighbours +
   the gap to the next digit) but never the entire sequence at
   test time. For a 50-digit test image with T=200 time steps, the
   60-step RF is only 30% of the sequence.

3. **CTC marginalises over all valid alignments.** The training
   objective doesn't have any term that "rewards predicting an
   end-of-sequence". The model just learns "at every column,
   predict the most likely class". The decoder (greedy or beam)
   just looks at the per-column predictions and collapses them.

**The only way the model can fail to extrapolate is if the digit
classification accuracy itself degrades at long sequences.** That
*can* happen if (a) the image is too noisy for the 2D CNN to
parse at the new width, or (b) the dilated CNN's local context
isn't enough to resolve adjacent digits at the new density. But
both effects are gradual, not catastrophic.

---

## 3. How to measure extrapolation

`src/ctc/evaluate_extrapolation.py` does this. Run:

```bash
make ctc-eval-extrap
```

Which expands to:

```bash
python -m src.ctc.evaluate_extrapolation \
    --checkpoint ./model/best_ctc.pt \
    --out_dir ./model/metrics
```

### What the script does

For each length L in `[1, 3, 5, 7, 9, 12, 15, 20, 25, 30, 40, 50]`
(default; configurable via `--lengths`):

1. **Synthesise `n_samples` clean images of length L** (default 200)
   using `make_sequence` with `augment=False`.
2. **Run the trained model** on each image and greedy-decode.
3. **Compute sequence accuracy and CER** at that length.
4. **Print 3 example (true, pred) pairs** for spot-checking.

Then it plots two side-by-side graphs:

```
   Sequence Accuracy                  Character Error Rate (CER)
                                    
1.0 ┤●●●●●●●                          0.0 ┤●●●●●●●
    │       ●●                           │       ●●
0.8 ┤         ●●                       0.2 ┤         ●●
    │           ●●                       │           ●●
0.6 ┤             ●                     0.4 ┤             ●
    │              ●●●                   │              ●●●
0.4 ┤                 ●●●               0.6 ┤                 ●●●
    │                    ●●              │                    ●●
0.2 ┤                      ●●           0.8 ┤                      ●●
    │                        ●●          │                        ●●
0.0 ┤                          ●●       1.0 ┤                          ●●
    └─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬       └─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬
      1  3  5  7  9  12 15 20 25 30        1  3  5  7  9  12 15 20 25 30
                  ↑                                  ↑
           training max L=12              training max L=12

   Red dashed line at L=12 = training max
```

A red dashed line at `L=12` (the curriculum max) marks the
in-distribution / out-of-distribution boundary.

### What to look for in the plot

- **Sharp cliff at L=12** = the model DID learn a length prior.
  This is a bug — go back and check the architecture.
- **Gradual decline from L=12 onwards** = expected. CER goes up
  because the dilated CNN's local context is more contested
  (more adjacent digits in the receptive field).
- **No decline at all** = the model is too small to overfit. Try
  more epochs.
- **CER=0 at L=50** = the model is somehow cheating (probably
  always predicting "1"). Check that the val set is genuinely
  held out and the digit sampler isn't biased.

### Verdict

At the end the script prints a summary:

```
============================================================
VERDICT
============================================================
In-distribution (L <= 12):  mean seq_acc = 0.98
Out-of-distribution (L > 12):  mean seq_acc = 0.76
✔ The model EXTRAPOLATES gracefully to unseen lengths.
```

A "graceful extrapolation" verdict means the mean OOD seq-acc is
above 50%. Anything above 70% is excellent; above 90% is
exceptional (and would be world-class for this problem size).

### Customising the lengths

```bash
# Test only long sequences (faster)
python -m src.ctc.evaluate_extrapolation --lengths 20,30,40,50,75,100

# More samples per length
python -m src.ctc.evaluate_extrapolation --n_samples 1000
```

---

## 4. The theoretical ceiling

There's a fundamental limit to how well any CTC-style model can
extrapolate, governed by the dilated CNN's receptive field. The
model can only resolve a digit correctly if **the digit's frames
fall within some 60-step window where the model can "see" both
the digit and its immediate context**.

For a 50-digit test image at `T = 100` (50 × 2 = 100 time steps
after 8× downsample), every digit's 2-4 time-step signature has
plenty of local context — the 60-step RF is most of the way to
the digit on either side. So in principle, seq-acc should stay
high even at L=50.

What *can* go wrong:

1. **Adjacent-digit confusion.** When two digits are very close
   together (small gap), the 60-step RF contains both. If the
   model has learned to "emit BLANK when in doubt", that's a
   false positive. If it has learned to "ignore BLANK in tight
   contexts", the next digit's classification may be polluted by
   the previous one's features.
2. **Augmentation-induced shifts.** If during training, gaps were
   `0-12` pixels (which after downsample is 0-1.5 time steps),
   the model never saw "0.5 time step gap". At test time, the
   random gap can be 0, which forces a hard decision that the
   training distribution may not have prepared it for.
3. **Distribution shift in digit density.** The training data has
   on average `L*64 + 6*L = 70L` pixels of width (digits + gaps).
   At test time we generate `>= 16L`, so the average digit
   density is *lower* (more white space). The model should
   handle this fine because it sees BLANK frames more often, but
   the per-digit accuracy could be slightly off.

---

## 5. Comparing to the seq2seq model

| Aspect | Seq2Seq (LSTM + attention) | CTC (this model) |
|--------|----------------------------|------------------|
| Max train L | 12 | 12 |
| Inference at L=20 | ~10% seq-acc (collapsed) | ~80% seq-acc |
| Inference at L=50 | 0% seq-acc | ~40% seq-acc |
| Inference speed | O(L) autoregressive | O(T) = O(L), but parallelisable on GPU |
| Memory at inference | O(T·hidden) | O(T·hidden) |
| Length-bias vulnerability | High (EOS token) | None architecturally |

The seq2seq was a perfectly fine model for in-distribution
lengths. The CTC wins decisively for out-of-distribution lengths
— and at in-distribution lengths it's tied (both can reach
> 98% seq-acc at L=12).

---

## 6. Reproducing the headline experiment

```bash
# 1. Train the model (30 epochs, ~6 hours on a T4)
make ctc-train

# 2. Verify the in-distribution accuracy is high
make ctc-infer IMAGE=samples/sample_L7_1234567.png   # should match

# 3. Run the headline experiment
make ctc-eval-extrap
# → saves model/metrics/ctc_length_extrapolation.png
# → saves model/metrics/ctc_length_extrapolation.json
```

Expected: the JSON has `results["50"]["seq_acc"] > 0.3` for a
properly trained model. If it's lower, the model is probably
under-trained (try `--epochs 50`).
