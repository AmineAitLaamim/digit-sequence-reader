"""
Length-extrapolation evaluation for the Seq2Seq digit-sequence-reader.

Mirrors src/ctc/evaluate_extrapolation.py exactly — same three metrics,
same output format, same plot layout — so results are directly comparable.

Metrics at each length L:

  digit_acc   -- % of individual digits decoded correctly (1 - CER).
                 Insensitive to sequence length; stays meaningful even at L=500.
  length_acc  -- % of sequences where the model emits EXACTLY the right
                 number of digits before EOS.  An autoregressive model with
                 a hard-wired length prior will collapse here for OOD lengths.
  seq_acc     -- % of sequences decoded with zero errors (every digit correct).
                 Kept for reference; naturally declines with L.

Key difference from CTC:
  The Seq2Seq decoder is AUTOREGRESSIVE — it stops when it predicts EOS,
  so length_acc directly measures whether the model has learned a length
  prior from training (max_seq_len_final = 12).  If it has, length_acc will
  collapse for L > 12 even though digit_acc might stay reasonable.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .config import config
from .model import Seq2Seq
from .dataset_aggressive import (
    build_multidigit_bank,
    get_digit_aug_pipeline,
    make_sequence,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _levenshtein(a, b):
    """Standard DP edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _greedy_decode_seq2seq(model, img_tensor, device):
    """
    Run one inference step and return the decoded digit list (ints 0-9).

    The Seq2Seq forward pass with targets=None runs until every batch
    element emits EOS, so we just strip SOS/EOS/PAD from the argmax.
    """
    with torch.no_grad():
        logits, _ = model(img_tensor.to(device), targets=None,
                          teacher_forcing_ratio=0.0)   # [1, L, vocab_size]

    preds = logits.argmax(dim=-1)[0].cpu().tolist()   # list of token ids

    digits = []
    for p in preds:
        if p == config['EOS_IDX']:
            break
        if p < 10:          # 0-9 are digit tokens
            digits.append(p)
    return digits


# ─────────────────────────────────────────────────────────────────────────────
# Per-length evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_at_length(model, bank, aug, L, n_samples, device):
    """
    Generate n_samples sequences of length L and compute three metrics:

      digit_acc  : fraction of individual digits decoded correctly  (1 - CER)
      length_acc : fraction of sequences where len(pred) == len(true)
      seq_acc    : fraction of sequences decoded with zero errors

    Returns (digit_acc, length_acc, seq_acc)
    """
    # Temporarily force the dataset to produce sequences of exactly length L.
    saved_min = config['min_seq_len']
    saved_max = config['max_seq_len']
    config['min_seq_len'] = L
    config['max_seq_len'] = L

    edit_dist      = 0
    total_chars    = 0
    length_correct = 0
    seq_correct    = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            img, label_tensor = make_sequence(bank, aug, config,
                                              augment=False, epoch=1)
            # label_tensor = [SOS, d0, d1, …, dL-1, EOS]
            true_digits = label_tensor[1:-1].tolist()   # strip SOS and EOS

            pred = _greedy_decode_seq2seq(model, img.unsqueeze(0), device)

            # ── Digit accuracy (via edit distance) ───────────────────────
            ed = _levenshtein(pred, true_digits)
            edit_dist   += ed
            total_chars += len(true_digits)

            # ── Length accuracy ───────────────────────────────────────────
            if len(pred) == len(true_digits):
                length_correct += 1

            # ── Sequence accuracy (zero-error, kept for reference) ────────
            if pred == true_digits:
                seq_correct += 1

    config['min_seq_len'] = saved_min
    config['max_seq_len'] = saved_max

    cer        = edit_dist / max(1, total_chars)
    digit_acc  = 1.0 - cer
    length_acc = length_correct / n_samples
    seq_acc    = seq_correct    / n_samples

    return digit_acc, length_acc, seq_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Seq2Seq model checkpoint (.pt)')
    parser.add_argument('--out_dir',    type=str, default='./metrics',
                        help='Where to save the plot & JSON')
    parser.add_argument('--lengths',    type=str,
                        default='5,12,20,50,75,100,150,200,300,400,500',
                        help='Comma-separated list of lengths to test')
    parser.add_argument('--n_samples',  type=int, default=100,
                        help='Number of synthetic sequences per length')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Seq2Seq().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # ── Build clean digit bank (no augmentation) ──────────────────────────
    print("Building digit bank (clean, no augmentation)...")
    bank      = build_multidigit_bank(config['data_path'])
    clean_aug = get_digit_aug_pipeline(augment=False, config=config)

    lengths   = [int(L) for L in args.lengths.split(',')]
    train_max = config.get('max_seq_len_final', 12)

    print(f"Testing lengths : {lengths}  ({args.n_samples} samples each)")
    print(f"Training max L  : {train_max}  (sequences beyond this are OOD)\n")

    # ── Header ────────────────────────────────────────────────────────────
    print(f"  {'L':>4}  {'digit_acc':>10}  {'length_acc':>11}  {'seq_acc':>9}  {'OOD':>4}")
    print("  " + "-" * 80)

    results = {}
    for L in lengths:
        digit_acc, length_acc, seq_acc = evaluate_at_length(
            model, bank, clean_aug, L, args.n_samples, device)

        results[L] = {
            'digit_acc':  digit_acc,
            'length_acc': length_acc,
            'seq_acc':    seq_acc,
        }

        ood_tag = " OOD" if L > train_max else "    "
        print(f"  {L:>4}  {digit_acc:>9.1%}  {length_acc:>10.1%}  "
              f"{seq_acc:>8.1%}  {ood_tag}")

    # ── Plot (3 panels) ───────────────────────────────────────────────────
    Ls          = sorted(results.keys())
    digit_accs  = [results[L]['digit_acc']  for L in Ls]
    length_accs = [results[L]['length_acc'] for L in Ls]
    seq_accs    = [results[L]['seq_acc']    for L in Ls]

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle(
        f"Seq2Seq Length Extrapolation  —  {os.path.basename(args.checkpoint)}",
        fontsize=13, y=1.02)

    def _vline(ax):
        ax.axvline(x=train_max, color='red', linestyle='--', linewidth=1.5,
                   label=f'Train max L={train_max}')
        ax.set_xlabel('Sequence length L')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # Panel 1 — Digit Accuracy
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(Ls, digit_accs, marker='o', linewidth=2, color='tab:green',
             label='Digit accuracy')
    ax1.set_ylabel('Digit Accuracy  (1 - CER)')
    ax1.set_title('Per-digit accuracy\n(robust to sequence length)')
    _vline(ax1)

    # Panel 2 — Length Accuracy
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(Ls, length_accs, marker='s', linewidth=2, color='tab:orange',
             label='Length accuracy')
    ax2.set_ylabel('Length Accuracy  (len_pred == len_true)')
    ax2.set_title('Output length accuracy\n(detects length-prior failure)')
    _vline(ax2)

    # Panel 3 — Sequence Accuracy (reference)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(Ls, seq_accs, marker='^', linewidth=2, color='tab:blue',
             label='Seq accuracy', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Sequence Accuracy  (all digits correct)')
    ax3.set_title('Full-sequence accuracy\n(reference — declines with length by design)')
    _vline(ax3)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'seq2seq_length_extrapolation.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved  : {plot_path}")
    plt.close()

    # ── JSON dump ─────────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, 'seq2seq_length_extrapolation.json')
    with open(json_path, 'w') as f:
        json.dump({
            'checkpoint':    os.path.basename(args.checkpoint),
            'n_samples':     args.n_samples,
            'train_max_len': train_max,
            'results': {
                str(L): v for L, v in results.items()
            },
        }, f, indent=2)
    print(f"JSON saved  : {json_path}")

    # ── Verdict ───────────────────────────────────────────────────────────
    in_dist = [results[L] for L in lengths if L <= train_max]
    ood     = [results[L] for L in lengths if L >  train_max]

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if in_dist:
        print(f"In-distribution  (L <= {train_max}):  "
              f"digit_acc={np.mean([r['digit_acc']  for r in in_dist]):.1%}  "
              f"length_acc={np.mean([r['length_acc'] for r in in_dist]):.1%}")
    if ood:
        mean_da = np.mean([r['digit_acc']  for r in ood])
        mean_la = np.mean([r['length_acc'] for r in ood])
        print(f"Out-of-distribution (L > {train_max}):  "
              f"digit_acc={mean_da:.1%}  "
              f"length_acc={mean_la:.1%}")

        if mean_da > 0.85 and mean_la > 0.80:
            print("=> The model EXTRAPOLATES well to unseen lengths.")
        elif mean_la < 0.50:
            print("=> Length accuracy collapse detected — model learned a length prior.")
        else:
            print("=> Graceful degradation (expected for very long sequences).")


if __name__ == '__main__':
    main()
