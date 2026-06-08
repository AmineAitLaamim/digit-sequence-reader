"""
Generate a single digit-sequence sample of a SPECIFIC length.

Usage:
    python -m src.ctc.generate_one --length 7 --out samples/test_L7.png
    python -m src.ctc.generate_one --length 25                          # auto-named file
    python -m src.ctc.generate_one --length 12 --augment                # with augmentation
    python -m src.ctc.generate_one --length 7 --count 5                 # 5 random samples of length 7
    python -m src.ctc.generate_one --length 7 --seed 42                 # reproducible

The width of the generated image is automatically chosen to satisfy the
CTC constraint: W >= max(64, L * width_per_digit), snapped to a multiple
of 8, so the downsampled time axis is at least 2*L.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

from .config import config
from .dataset import build_multidigit_bank, get_digit_aug_pipeline, make_sequence


# Module-level cache for the digit bank + augmentation pipeline.
# Building the bank (EMNIST + QMNIST + USPS) takes ~30s the first time;
# caching makes --count N essentially instantaneous after the first sample.
_BANK_CACHE      = {}
_AUG_CACHE       = {}


def _get_bank_and_aug(augment, epoch):
    key = (augment, epoch)
    if key not in _BANK_CACHE:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data')
        if 'bank' not in _BANK_CACHE:
            print(f"Building digit bank from {data_path} ...")
            _BANK_CACHE['bank'] = build_multidigit_bank(data_path)
        _BANK_CACHE[key] = (_BANK_CACHE['bank'],
                            get_digit_aug_pipeline(augment=augment,
                                                    config=config, epoch=epoch))
    return _BANK_CACHE[key]


def generate_one(length, augment=False, out_path=None, epoch=1, seed=None):
    """Generate and save a single sample of `length` digits.

    Returns: (out_path, digit_string)
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    bank, aug = _get_bank_and_aug(augment, epoch)

    # Force make_sequence to use this exact length by patching the config.
    saved_min = config['min_seq_len']
    saved_max = config['max_seq_len']
    config['min_seq_len'] = length
    config['max_seq_len'] = length

    img_tensor, digits = make_sequence(bank, aug, config,
                                       augment=augment, epoch=epoch)

    # Restore config so subsequent calls / other code see the original.
    config['min_seq_len'] = saved_min
    config['max_seq_len'] = saved_max

    digit_str = "".join(map(str, digits))
    W = img_tensor.shape[2]
    T = W // 8
    print(f"Generated: length={len(digits)}  width={W}  T={T}  T/L={T / len(digits):.2f}")
    print(f"Digits:    {digit_str}")

    img_pil = transforms.ToPILImage()(img_tensor)
    if out_path is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'samples')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sample_L{length}_{digit_str}.png")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    img_pil.save(out_path)
    print(f"Saved to: {out_path}")
    return out_path, digit_str


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single digit-sequence sample of a specific length.")
    parser.add_argument('--length',  type=int, required=True,
                        help='Number of digits in the sequence')
    parser.add_argument('--out',     type=str, default=None,
                        help='Output PNG path (default: ./samples/sample_L<length>_<digits>.png)')
    parser.add_argument('--count',   type=int, default=1,
                        help='How many samples to generate (each gets a different random digit string)')
    parser.add_argument('--augment', action='store_true',
                        help='Apply training-style augmentation (default: clean)')
    parser.add_argument('--seed',    type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.length < 1:
        parser.error("--length must be >= 1")

    if args.count == 1:
        generate_one(
            length=args.length,
            augment=args.augment,
            out_path=args.out,
            seed=args.seed,
        )
    else:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'samples')
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i in range(args.count):
            # Vary the seed per iteration (unless user pinned one) so each
            # call produces a different random digit string.
            iter_seed = None if args.seed is None else args.seed + i
            out_path, _ = generate_one(
                length=args.length,
                augment=args.augment,
                out_path=None,
                seed=iter_seed,
            )
            paths.append(out_path)
        print(f"\nGenerated {args.count} samples of length {args.length}:")
        for p in paths:
            print(f"  {p}")


if __name__ == '__main__':
    main()
