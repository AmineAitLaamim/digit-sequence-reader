"""
CRNN + CTC training script for digit-sequence-reader.

Usage (e.g. on Colab):
    !python -m src.ctc.train --drive_path /content/drive/MyDrive/digit-sequence-reader
    !python -m src.ctc.train --drive_path ./ --epochs 30 --batch_size 64 --lr 1e-3
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import config
from .model import CRNN_CTC, greedy_decode
from .dataset import (
    InfiniteCTCDataset,
    collate_fn as ctc_collate_fn,
    get_dataloaders,
)


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────
def compute_metrics(decoded, target_lengths, targets_flat):
    """
    Args:
        decoded:        list[list[int]]   - predictions from greedy_decode
        target_lengths: tensor [B]        - number of digits per item
        targets_flat:   tensor [sum(L_i)] - 1D flattened ground truth
    Returns:
        (seq_acc, cer)  - sequence accuracy and character error rate
    """
    B = len(decoded)
    seq_correct = 0
    edit_distance = 0
    total_chars = 0

    # Reconstruct the per-item ground-truth list from the 1D flattened tensor.
    cursor = 0
    for i, length in enumerate(target_lengths.tolist()):
        gt = targets_flat[cursor: cursor + length].tolist()
        cursor += length

        pred = decoded[i]
        # Sequence accuracy: exact match.
        if pred == gt:
            seq_correct += 1
        # CER: Levenshtein edit distance / number of GT characters.
        edit_distance += _levenshtein(pred, gt)
        total_chars   += length

    seq_acc = seq_correct / B
    cer     = edit_distance / max(1, total_chars)
    return seq_acc, cer


def _levenshtein(a, b):
    """Standard dynamic-programming edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),   # substitution
            )
        prev = cur
    return prev[-1]


# ────────────────────────────────────────────────────────────────────────────
# Train / validation loops
# ────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, device, steps_per_epoch, log_every=50):
    model.train()
    total_loss  = 0.0
    total_seq   = 0.0
    total_cer   = 0.0
    n_batches   = 0

    pbar = tqdm(enumerate(dataloader), total=steps_per_epoch,
                desc="  train", leave=False)
    for i, batch in pbar:
        if i >= steps_per_epoch:
            break

        images         = batch['images'].to(device)
        targets        = batch['targets'].to(device)
        # NB: `batch['input_lengths']` is intentionally NOT read here.
        # The model computes `input_lengths` internally from the actual
        # CNN output width (T = feat.size(3)), so passing it would be
        # redundant. The collate_fn still returns it for inspection /
        # future CTC use-cases (e.g. per-sample variable-T inputs).
        target_lengths = batch['target_lengths'].to(device)

        optimizer.zero_grad()
        # Forward returns (loss, logits) when targets are given — the
        # logits come for free, saving us a second forward pass for
        # metric logging.
        loss, logits = model(images, targets=targets, target_lengths=target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        optimizer.step()

        total_loss += loss.item()

        # Compute metrics (greedy decode on every log_every steps for speed).
        if (i + 1) % log_every == 0 or (i + 1) == steps_per_epoch:
            with torch.no_grad():
                # No second forward pass — logits were already returned
                # by the training forward. We detach() to free the autograd graph.
                decoded = greedy_decode(logits.detach())
                seq_acc, cer = compute_metrics(
                    decoded, target_lengths, targets)
            total_seq += seq_acc
            total_cer += cer
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             seq=f"{seq_acc:.3f}",
                             cer=f"{cer:.3f}")
        else:
            # Track loss only between metric logs
            pass

    avg_loss = total_loss / steps_per_epoch
    avg_seq  = total_seq / max(1, n_batches)
    avg_cer  = total_cer / max(1, n_batches)
    return avg_loss, avg_seq, 1.0 - avg_cer   # return char accuracy = 1 - CER


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_seq  = 0.0
    total_cer  = 0.0
    n_batches  = 0

    for batch in tqdm(dataloader, desc="  val  ", leave=False):
        images         = batch['images'].to(device)
        targets        = batch['targets'].to(device)
        # (See train_one_epoch for why batch['input_lengths'] is unused.)
        target_lengths = batch['target_lengths'].to(device)

        # Forward returns (loss, logits) when targets are given. We use
        # the free logits for metric decoding — no second forward needed.
        loss, logits = model(images, targets=targets, target_lengths=target_lengths)
        total_loss += loss.item()

        # Decode for metrics.
        decoded = greedy_decode(logits)
        seq_acc, cer = compute_metrics(decoded, target_lengths, targets)

        total_seq += seq_acc
        total_cer += cer
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_seq  = total_seq  / n_batches
    avg_cer  = total_cer  / n_batches
    return avg_loss, avg_seq, 1.0 - avg_cer


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, val_seq_acc, path):
    torch.save({
        'epoch':        epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':     val_loss,
        'val_seq_acc':  val_seq_acc,
        'config':       config,
    }, path)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_path', type=str, required=True,
                        help='Base folder for checkpoints / metrics')
    parser.add_argument('--epochs',     type=int, default=config['epochs'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--lr',         type=float, default=config['lr'])
    parser.add_argument('--resume',     type=str, default='',
                        help='Path to best_ctc.pt to resume from')
    args = parser.parse_args()

    # Override config with CLI arguments
    config['epochs']     = args.epochs
    config['batch_size'] = args.batch_size
    config['lr']         = args.lr
    config['drive_path'] = args.drive_path

    # Keep the CTC checkpoints in a dedicated folder so they never
    # overwrite the seq2seq ones.
    checkpoint_dir = os.path.join(args.drive_path, 'checkpoints_ctc')
    metrics_dir    = os.path.join(args.drive_path, 'metrics')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir,    exist_ok=True)

    metrics_file = os.path.join(metrics_dir, 'ctc_metrics.csv')
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_seq_acc', 'val_seq_acc',
                'train_char_acc', 'val_char_acc', 'lr',
            ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Building digit bank and validation/test loaders...")
    digit_bank, val_loader, test_loader = get_dataloaders(
        data_path=config['data_path'])
    steps_per_epoch = config['train_size'] // config['batch_size']
    print(f"Steps per epoch: {steps_per_epoch}")

    # Model + optimizer
    model     = CRNN_CTC().to(device)
    optimizer = AdamW(model.parameters(),
                      lr=config['lr'],
                      weight_decay=config.get('weight_decay', 1e-4))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['lr_patience'],
        factor=config['lr_factor'],
        min_lr=config['lr_min'],
    )

    # Optional resume
    start_epoch   = 1
    best_val_seq  = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass
        start_epoch  = ckpt['epoch'] + 1
        best_val_seq = ckpt.get('val_seq_acc', 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} | val_seq_acc={best_val_seq:.4f}")

    early_stop_counter = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} trainable parameters")

    for epoch in range(start_epoch, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        # Train loader: rebuild each epoch so the curriculum updates aug.
        train_ds = InfiniteCTCDataset(
            digit_bank, config, size=config['train_size'],
            augment=True, epoch=epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            collate_fn=ctc_collate_fn,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
            persistent_workers=True,
        )

        train_loss, train_seq, train_char = train_one_epoch(
            model, train_loader, optimizer, device, steps_per_epoch)
        val_loss,   val_seq,   val_char   = validate(
            model, val_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train | loss={train_loss:.4f} seq_acc={train_seq:.4f} char_acc={train_char:.4f}")
        print(f"  Val   | loss={val_loss:.4f} seq_acc={val_seq:.4f} char_acc={val_char:.4f} | lr={current_lr}")

        scheduler.step(val_loss)

        if val_seq > best_val_seq:
            best_val_seq = val_seq
            early_stop_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_seq,
                path=os.path.join(checkpoint_dir, 'best_ctc.pt'))
            print(f"  ✔ New best model saved (val_seq_acc={val_seq:.4f})")
        else:
            early_stop_counter += 1

        with open(metrics_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_seq, val_seq,
                train_char, val_char, current_lr,
            ])

        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Final test-set evaluation
    print("\nLoading best checkpoint for test-set evaluation...")
    ckpt = torch.load(os.path.join(checkpoint_dir, 'best_ctc.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    test_loss, test_seq, test_char = validate(model, test_loader, device)
    print(f"  Test | loss={test_loss:.4f} seq_acc={test_seq:.4f} char_acc={test_char:.4f}")


if __name__ == '__main__':
    main()
