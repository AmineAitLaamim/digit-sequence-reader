"""
CTC dataset and collate function for the digit-sequence-reader.

Key differences from the autoregressive Seq2Seq version:
    * Targets are RAW digit lists (no SOS / EOS). CTC handles alignment
      via the BLANK token, so explicit start/end markers are unnecessary
      and would actually harm the model.
    * The collate function returns the exact 4-key dictionary that
      `nn.CTCLoss` needs:
          images           - padded image batch [B, 1, 64, max_w]
          targets          - 1D flattened target tensor [sum(L_i)]
          input_lengths    - [B] tensor, each entry == max_w // 8
          target_lengths   - [B] tensor, the EXACT number of digits per item
    * `input_lengths` uses the *batch-max* time length (`max_w // 8`),
      which equals the actual T of the CNN output for that batch element
      (the model is convolutional and "sees" the full padded width; the
      zero-padded region simply contributes empty features that CTC can
      label as BLANK).
"""

import os
import sys
import random
import ssl

# Bypass SSL verification for torchvision downloads (USPS specifically)
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import IterableDataset, DataLoader

from .config import config


# ────────────────────────────────────────────────────────────────────────────
# Digit bank (multi-source)
# ────────────────────────────────────────────────────────────────────────────
def build_multidigit_bank(data_path='./data'):
    """Load EMNIST Digits + QMNIST + USPS and merge them into a digit bank."""
    os.makedirs(data_path, exist_ok=True)
    bank = {i: [] for i in range(10)}

    print("Loading EMNIST Digits...")
    emnist = datasets.EMNIST(root=data_path, split='digits', train=True, download=True)
    for img, label in emnist:
        # EMNIST inherits the NIST SD-19 scanning orientation: digits are stored
        # rotated 90° counter-clockwise and mirrored horizontally relative to how
        # a human writes them.  The standard fix is to rotate 90° CW (== transpose
        # + flip) so the digit appears upright before it enters the augmentation
        # pipeline.  QMNIST and USPS do NOT need this correction.
        img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        bank[label].append(img)

    print("Loading QMNIST...")
    qmnist = datasets.QMNIST(root=data_path, what='train', compat=True, download=True)
    for img, label in qmnist:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        bank[label].append(img)

    print("Loading USPS...")
    usps = datasets.USPS(root=data_path, train=True, download=True)
    for img, label in usps:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = img.resize((28, 28), Image.BILINEAR)
        bank[label].append(img)

    for i in range(10):
        print(f"  Digit {i}: {len(bank[i])} images")
    return bank


# ────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ────────────────────────────────────────────────────────────────────────────
def get_digit_aug_pipeline(augment=True, config=None, epoch=1):
    """Same curriculum-style augmentation as the seq2seq version, but
    configurable for CTC use."""
    if config is None:
        config = globals()['config']

    if not augment:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    intensity = min(1.0, epoch / config.get('aug_warmup_epochs', 10))
    noise_var = config.get('aug_noise_var', (5, 20))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(A, 'GaussianNoise'):
            noise = A.GaussianNoise(std_range=tuple(v * intensity for v in noise_var))
        else:
            try:
                noise = A.GaussNoise(std_range=tuple(v * intensity for v in noise_var))
            except Exception:
                noise = A.GaussNoise(var_limit=(noise_var[0]**2 * intensity,
                                                noise_var[1]**2 * intensity))

    # Read rotation / shear from config (kept mild so digits stay
    # human-readable). If config has 0, we skip the transform entirely
    # to keep the augmentation pipeline minimal.
    rot_limit  = int(config.get('seq_rotation', 0))            # default 0 → no rotation
    shear_lim  = int(config.get('seq_shear', 0))              # default 0 → no shear
    geo_steps  = []
    if rot_limit > 0:
        # Scaled by `intensity` so early epochs get less rotation.
        geo_steps.append(A.Affine(
            rotate=(-rot_limit * intensity, rot_limit * intensity),
            shear=(-shear_lim * intensity, shear_lim * intensity) if shear_lim > 0 else 0,
            translate_percent=0,
            scale=(1.0, 1.0),
            p=0.5,
        ))

    transform = A.Compose(
        geo_steps + [
            A.RandomBrightnessContrast(
                brightness_limit=config.get('aug_brightness', 0.3) * intensity,
                contrast_limit=config.get('aug_contrast', 0.3) * intensity,
                p=0.5),
            A.OneOf([
                noise,
                A.MotionBlur(blur_limit=max(3, int(config.get('aug_blur_limit', 3) * intensity)))
            ], p=0.4 * intensity),
            A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                p=config.get('aug_erasing_p', 0.3) * intensity),
            A.Resize(64, 64),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def apply(pil_image):
        np_img = np.array(pil_image, dtype=np.uint8)
        return transform(image=np_img)['image']

    return apply


# ────────────────────────────────────────────────────────────────────────────
# Sequence generation with WIDTH GUARANTEE
# ────────────────────────────────────────────────────────────────────────────
def get_curriculum_max_len(epoch, config):
    """Same length curriculum as the seq2seq version."""
    base   = config['max_seq_len']
    final  = config.get('max_seq_len_final', 12)
    warmup = config.get('aug_warmup_epochs', 10)
    if epoch <= warmup:
        return base
    progress = min(1.0, (epoch - warmup) / 20.0)
    return int(base + progress * (final - base))


def make_sequence(digit_bank, aug_pipeline, config, augment=False, epoch=1):
    """
    Build a (image, digit_list) pair.

    CRITICAL: The image width is forced to satisfy
            W >= max(min_image_width, num_digits * width_per_digit)
    so that after the 8x CNN downsample, T = W // 8 is at least
    `2 * num_digits` — giving CTC enough blank slots to place
    each digit correctly.
    """
    max_len = get_curriculum_max_len(epoch, config) if augment else config['max_seq_len']
    L = random.randint(config['min_seq_len'], max_len)
    digits = [random.randint(0, 9) for _ in range(L)]

    # Augmentation / overlap intensity
    aug_intensity  = min(1.0, epoch / config.get('aug_warmup_epochs', 10))
    if augment and epoch > config.get('overlap_start_epoch', 5):
        overlap_intensity = min(1.0, (epoch - config.get('overlap_start_epoch', 5)) / 10.0)
        overlap_prob      = config.get('overlap_prob_max', 0.10) * overlap_intensity
    else:
        overlap_prob = 0.0

    # ── ⭐ CTC width guarantee ⭐ ─────────────────────────────────────
    # Required minimum width: max(64, L * 16) → T >= 2 * L after downsample.
    min_w = max(config['min_image_width'],
                L * config['width_per_digit'])
    # Pick a slightly-random width above the minimum so the model
    # doesn't see the same canvas for the same L every time.
    target_w = min_w + random.randint(0, max(0, min_w // 2))
    # Snap to a multiple of 8 so W // 8 is exact.
    target_w = ((target_w + 7) // 8) * 8

    sequence_parts = []
    consumed_w = 0
    for i, digit in enumerate(digits):
        img_pil = random.choice(digit_bank[digit])
        img_tensor = aug_pipeline(img_pil)         # [1, 64, 64]

        if i > 0:
            if augment and random.random() < overlap_prob:
                gap = -random.randint(1, config.get('overlap_max', 8))
            else:
                gap = random.randint(config.get('gap_min', 0),
                                     config.get('gap_max', 12))

            if gap >= 0:
                spacer = torch.zeros(1, 64, gap)
                sequence_parts.append(spacer)
                consumed_w += gap
            elif gap < 0:
                if sequence_parts:
                    sequence_parts[-1] = sequence_parts[-1][:, :, :-abs(gap)]
                    consumed_w -= abs(gap)

        sequence_parts.append(img_tensor)
        consumed_w += 64

    seq_img = torch.cat(sequence_parts, dim=2)      # [1, 64, W]

    # Pad the right side out to `target_w` with zeros.
    if seq_img.size(2) < target_w:
        right_pad = target_w - seq_img.size(2)
        seq_img = torch.nn.functional.pad(seq_img, (0, right_pad), value=0)
    # If random gaps pushed us over the target, that's fine — width can vary.

    # CTC expects raw digits, no SOS / EOS.
    return seq_img, digits


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────
class InfiniteCTCDataset(IterableDataset):
    """Streams (image, digit_list) samples using the multi-digit bank."""

    def __init__(self, digit_bank, config, size=None, augment=True, epoch=1):
        super().__init__()
        self.digit_bank  = digit_bank
        self.config      = config
        self.size        = size
        self.augment     = augment
        self.epoch       = epoch
        self.aug_pipeline = get_digit_aug_pipeline(
            augment=self.augment, config=self.config, epoch=self.epoch)

    def __len__(self):
        if self.size is None:
            raise TypeError("Dataset has no fixed size.")
        return self.size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_size = self.size

        if worker_info is not None:
            np.random.seed(np.random.get_state()[1][0] + worker_info.id)
            random.seed(random.getstate()[1][0] + worker_info.id)

            if self.size is not None:
                worker_size = self.size // worker_info.num_workers
                if worker_info.id == worker_info.num_workers - 1:
                    worker_size += self.size % worker_info.num_workers

        count = 0
        while True:
            if worker_size is not None and count >= worker_size:
                break
            yield make_sequence(self.digit_bank, self.aug_pipeline,
                                self.config, augment=self.augment, epoch=self.epoch)
            count += 1


# ────────────────────────────────────────────────────────────────────────────
# Collate function (THE 4-KEY DICTIONARY REQUIRED BY nn.CTCLoss)
# ────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    """
    Returns a dict with EXACTLY these 4 keys:
        - 'images'         [B, 1, 64, max_w]     padded image batch
        - 'targets'        [sum(L_i)]            1D flattened digits
        - 'input_lengths'  [B]                   == max_w // 8 for each
        - 'target_lengths' [B]                   exact digit count per item
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]   # list[list[int]]

    B = len(batch)
    H = config['img_height']
    max_w = max(img.size(2) for img in images)

    # Pad images on the right (zero is consistent with background).
    padded_images = torch.zeros(B, 1, H, max_w)
    for i, img in enumerate(images):
        w = img.size(2)
        padded_images[i, :, :, :w] = img

    # Flatten target digit lists (NO SOS/EOS) into a single 1D tensor.
    flat_targets = []
    target_lengths = []
    for tgt in targets:
        flat_targets.extend(tgt)
        target_lengths.append(len(tgt))

    targets_tensor      = torch.tensor(flat_targets, dtype=torch.long)
    target_lengths      = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths       = torch.full((B,), max_w // 8, dtype=torch.long)

    return {
        'images':         padded_images,        # [B, 1, 64, max_w]
        'targets':        targets_tensor,       # [sum(L_i)]
        'input_lengths':  input_lengths,        # [B]
        'target_lengths': target_lengths,       # [B]
    }


# ────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ────────────────────────────────────────────────────────────────────────────
def get_dataloaders(data_path=None):
    """Returns (digit_bank, val_loader, test_loader).

    The train loader is recreated each epoch in `train.py` so the
    augmentation intensity curriculum can be updated.
    """
    data_path = data_path or config['data_path']
    bank = build_multidigit_bank(data_path)

    val_ds  = InfiniteCTCDataset(bank, config, size=config['val_size'],  augment=False)
    test_ds = InfiniteCTCDataset(bank, config, size=config['test_size'], augment=False)

    num_workers = config.get('num_workers', 2)
    val_loader  = DataLoader(val_ds,  batch_size=config['batch_size'],
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True)
    return bank, val_loader, test_loader


# ────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building digit bank (this may take a minute on first run)...")
    bank, val_loader, test_loader = get_dataloaders()
    batch = next(iter(val_loader))
    print("Batch keys         :", list(batch.keys()))
    print("images shape       :", tuple(batch['images'].shape))
    print("targets shape      :", tuple(batch['targets'].shape))
    print("input_lengths shape:", tuple(batch['input_lengths'].shape))
    print("target_lengths     :", batch['target_lengths'].tolist())
    assert set(batch.keys()) == {'images', 'targets', 'input_lengths', 'target_lengths'}
    # Sanity: T >= L for every item in the batch (CTC requirement)
    for i, (T, L) in enumerate(zip(batch['input_lengths'], batch['target_lengths'])):
        assert T.item() >= L.item(), f"item {i}: T={T.item()} < L={L.item()} → CTC would fail"
    print("All CTC length constraints satisfied ✔")
