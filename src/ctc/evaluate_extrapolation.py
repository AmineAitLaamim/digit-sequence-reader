"""
Length-extrapolation evaluation for the CRNN + CTC digit-sequence-reader.

This script is the *definitive test* that the CTC model is structurally
incapable of learning a length prior. We synthesise digit sequences of
lengths that the model NEVER saw during training (the training distribution
only goes up to `max_seq_len_final = 12`), and measure both sequence
accuracy and CER at each length.

A working CTC model should retain > 95% sequence accuracy at L=12 and
gracefully degrade (not catastrophically collapse) for L > 12.
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import numpy as np

from .config import config
from .model import CRNN_CTC, greedy_decode
from .dataset import (
    build_multidigit_bank,
    get_digit_aug_pipeline,
    make_sequence,
)
from .train import _levenshtein  # re-use the metric


def evaluate_at_length(model, bank, aug, L, n_samples, device):
    """Generate n_samples sequences of length L and measure seq-acc + CER."""
    saved_min = config['min_seq_len']
    saved_max = config['max_seq_len']
    config['min_seq_len'] = L
    config['max_seq_len'] = L

    seq_correct = 0
    edit_dist   = 0
    total_chars = 0
    examples    = []   # store a few (true, pred) pairs for inspection

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            img, true_digits = make_sequence(bank, aug, config, augment=False, epoch=1)
            logits = model(img.unsqueeze(0).to(device))   # [1, T, V]
            pred = greedy_decode(logits)[0]

            if pred == true_digits:
                seq_correct += 1
            edit_dist += _levenshtein(pred, true_digits)
            total_chars += len(true_digits)

            if len(examples) < 3:
                examples.append((true_digits, pred))

    # Restore
    config['min_seq_len'] = saved_min
    config['max_seq_len'] = saved_max

    seq_acc = seq_correct / n_samples
    cer     = edit_dist / max(1, total_chars)
    return seq_acc, cer, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt (CRNN + CTC checkpoint)')
    parser.add_argument('--out_dir',    type=str, default='./metrics',
                        help='Where to save the extrapolation plot & JSON')
    parser.add_argument('--lengths',    type=str, default='1,3,5,7,9,12,15,20,25,30,40,50',
                        help='Comma-separated list of lengths to test')
    parser.add_argument('--n_samples',  type=int, default=200,
                        help='Number of synthetic sequences per length')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CRNN_CTC().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # Build digit bank (clean pipeline, no aug)
    print("Building digit bank (clean, no augmentation)...")
    bank = build_multidigit_bank(config['data_path'])
    clean_aug = get_digit_aug_pipeline(augment=False, config=config)

    lengths = [int(L) for L in args.lengths.split(',')]
    print(f"Testing lengths: {lengths}  ({args.n_samples} samples each)\n")

    results = {}
    for L in lengths:
        seq_acc, cer, examples = evaluate_at_length(
            model, bank, clean_aug, L, args.n_samples, device)
        results[L] = {'seq_acc': seq_acc, 'cer': cer}
        # Print a few examples for spot-checking
        sample_str = " | ".join(
            f"true={''.join(map(str, t))} pred={''.join(map(str, p))}"
            for t, p in examples)
        print(f"  L={L:>2} | seq_acc={seq_acc:.3f} | cer={cer:.3f}  | {sample_str}")

    # ── Plot ─────────────────────────────────────────────────────────
    Ls       = sorted(results.keys())
    seq_accs = [results[L]['seq_acc'] for L in Ls]
    cers     = [results[L]['cer']     for L in Ls]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(Ls, seq_accs, marker='o', linewidth=2, color='tab:blue')
    ax1.axvline(x=config.get('max_seq_len_final', 12), color='red',
                linestyle='--', label=f"Train max L = {config.get('max_seq_len_final', 12)}")
    ax1.set_xlabel('Sequence length L')
    ax1.set_ylabel('Sequence Accuracy')
    ax1.set_title('CTC: Length Extrapolation (Sequence Acc)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(Ls, cers, marker='s', linewidth=2, color='tab:red')
    ax2.axvline(x=config.get('max_seq_len_final', 12), color='red',
                linestyle='--', label=f"Train max L = {config.get('max_seq_len_final', 12)}")
    ax2.set_xlabel('Sequence length L')
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_title('CTC: Length Extrapolation (CER)')
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'ctc_length_extrapolation.png')
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    # JSON dump for archival
    json_path = os.path.join(args.out_dir, 'ctc_length_extrapolation.json')
    with open(json_path, 'w') as f:
        json.dump({
            'checkpoint':    os.path.basename(args.checkpoint),
            'n_samples':     args.n_samples,
            'train_max_len': config.get('max_seq_len_final', 12),
            'results':       {str(L): v for L, v in results.items()},
        }, f, indent=2)
    print(f"JSON saved to {json_path}")

    # Summary verdict
    train_max = config.get('max_seq_len_final', 12)
    in_dist   = [results[L] for L in lengths if L <= train_max]
    ood       = [results[L] for L in lengths if L >  train_max]
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if in_dist:
        print(f"In-distribution (L <= {train_max}):  "
              f"mean seq_acc = {np.mean([r['seq_acc'] for r in in_dist]):.3f}")
    if ood:
        print(f"Out-of-distribution (L > {train_max}):  "
              f"mean seq_acc = {np.mean([r['seq_acc'] for r in ood]):.3f}")
    if ood and np.mean([r['seq_acc'] for r in ood]) > 0.5:
        print("✔ The model EXTRAPOLATES gracefully to unseen lengths.")
    elif ood:
        print("⚠ Some degradation outside the training distribution (expected for very long sequences).")
    else:
        print("(All tested lengths were within the training distribution.)")


if __name__ == '__main__':
    main()
