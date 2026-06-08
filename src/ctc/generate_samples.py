"""
CTC sample generator.

Generates images and writes them to disk for visual inspection.
Uses the same `make_sequence` from `dataset.py` so the width guarantee
(> num_digits * width_per_digit) is enforced.
"""

import os
import sys
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))

from .config import config
from .dataset import build_multidigit_bank, get_digit_aug_pipeline, make_sequence


def main():
    print("Downloading EMNIST, QMNIST, USPS and building digit bank...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(project_root, 'data')
    digit_bank = build_multidigit_bank(data_path)

    # Use max epoch to see the actual training distribution.
    epoch = config.get('epochs', 30)
    aug_pipeline = get_digit_aug_pipeline(augment=True, config=config, epoch=epoch)

    out_dir = os.path.join(project_root, 'samples')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating 20 CTC sample images at curriculum epoch {epoch}...")
    for i in range(20):
        img_tensor, digits = make_sequence(digit_bank, aug_pipeline, config,
                                           augment=True, epoch=epoch)
        digit_str = "".join(map(str, digits))

        img = transforms.ToPILImage()(img_tensor)
        filename = os.path.join(out_dir, f"ctc_sample_{i+1}_{digit_str}.png")
        img.save(filename)
        print(f"  Saved {filename}  (width={img_tensor.shape[2]}, digits={len(digits)}, "
              f"T={img_tensor.shape[2] // 8} >= 2*L={2*len(digits)})")


if __name__ == "__main__":
    main()
