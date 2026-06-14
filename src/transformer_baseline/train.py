"""
Transformer Seq2Seq training script.

Usage:
    # From repo root (Colab or local):
    !python -m src.transformer_baseline.train \\
        --checkpoint_dir /content/drive/MyDrive/digit-sequence-reader/checkpoint_transformer_baseline \\
        --epochs 30 --batch_size 64 --lr 1e-3 --max_length 12

Design notes:
    * IDENTICAL data pipeline to src/ctc/train.py — same dataset, same config,
      same augmentation schedule.  The ONLY difference is the model and the
      loss (CrossEntropy instead of CTC).
    * LR warmup is CRITICAL for Transformer stability (warmup_steps=2000).
      Without warmup the Adam adaptive learning rate starts with very large
      updates for parameters that have near-zero gradients, causing divergence
      in the first epoch.
    * All other hyperparameters (batch, clip_grad, early_stop_patience) are
      kept identical to the CTC baseline for a perfectly fair comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

# Add repo root to path so src.* imports resolve correctly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transformer_baseline.model import TransformerSeq2Seq
# Reusing existing dataset/utils to maintain DRY principles and ensure
# identical data pipeline for fair comparison.
from src.ctc.dataset import (
    InfiniteCTCDataset,
    collate_fn as ctc_collate_fn,
    get_dataloaders,
)
from src.ctc.config import config   # shared config dict

def transformer_collate_fn(batch):
    """Wraps CTC collate to return 2D padded targets for the Transformer."""
    ctc_dict = ctc_collate_fn(batch)
    
    # ctc_collate_fn returns 'targets' as 1D flat tensor and 'target_lengths'
    B = ctc_dict['target_lengths'].size(0)
    max_L = int(ctc_dict['target_lengths'].max().item())
    
    # 10 is PAD_IDX in our vocabulary
    targets_2d = torch.full((B, max_L), 10, dtype=torch.long)
    
    cursor = 0
    for i, L in enumerate(ctc_dict['target_lengths'].tolist()):
        targets_2d[i, :L] = ctc_dict['targets'][cursor: cursor + L]
        cursor += L
        
    ctc_dict['targets'] = targets_2d
    return ctc_dict


# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein distance (for CER computation)
# ─────────────────────────────────────────────────────────────────────────────
def _levenshtein(a: list, b: list) -> int:
    """Standard DP edit distance between two integer sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,              # deletion
                cur[j - 1] + 1,           # insertion
                prev[j - 1] + (ca != cb), # substitution
            )
        prev = cur
    return prev[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    decoded: list[list[int]],
    target_lengths: torch.Tensor,
    targets: torch.Tensor, # Can be 1D flat or 2D padded
) -> tuple[float, float]:
    """Compute sequence accuracy and character error rate. Handles both 1D and 2D targets."""
    B = len(decoded)
    seq_correct   = 0
    edit_distance = 0
    total_chars   = 0

    # Check if targets are 1D (flat CTC format) or 2D (padded Transformer format)
    is_1d = (targets.dim() == 1)
    cursor = 0

    for i, length in enumerate(target_lengths.tolist()):
        length = int(length)
        
        if is_1d:
            # 1D flat tensor (from default val_loader)
            gt = targets[cursor:cursor + length].tolist()
            cursor += length
        else:
            # 2D padded tensor (from transformer_collate_fn)
            gt = targets[i, :length].tolist()

        pred = decoded[i]

        if pred == gt:
            seq_correct += 1
        edit_distance += _levenshtein(pred, gt)
        total_chars   += length

    seq_acc = seq_correct / max(1, B)
    cer     = edit_distance / max(1, total_chars)
    return seq_acc, cer


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineLR:
    """Manual LR scheduler: linear warmup then cosine annealing to eta_min.

    CRITICAL for Transformer stability.
    Without warmup, the Adam adaptive learning rate starts with very large
    updates on near-zero gradients (especially in the token embeddings and
    cross-attention layers), causing divergence in the first epoch.

    Args:
        optimizer:     The wrapped AdamW optimizer.
        warmup_steps:  Number of steps for linear warmup (recommended: 2000).
        total_steps:   Total training steps (epochs * steps_per_epoch).
        base_lr:       Peak learning rate reached at end of warmup.
        eta_min:       Minimum LR at end of cosine decay (default: 1e-6).
    """

    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.base_lr      = base_lr
        self.eta_min      = eta_min
        self._step        = 0

    def step(self) -> None:
        """Update optimizer LR for the current global step."""
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_lr(self) -> float:
        s = self._step
        if s <= self.warmup_steps:
            # Linear ramp-up.
            return self.base_lr * s / max(1, self.warmup_steps)
        # Cosine decay after warmup.
        progress = (s - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min + (self.base_lr - self.eta_min) * cosine

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    model: TransformerSeq2Seq,
    optimizer: AdamW,
    epoch: int,
    val_loss: float,
    val_seq_acc: float,
    path: str,
) -> None:
    """Persist model + optimiser state to disk."""
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':             val_loss,
            'val_seq_acc':          val_seq_acc,
        },
        path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Full validation pass using greedy autoregressive decoding.

    Returns:
        (avg_loss, avg_seq_acc, avg_char_acc)
    """
    model.eval()
    total_loss = total_seq = total_cer = 0.0
    n_batches  = 0

    for batch in tqdm(dataloader, desc='  val  ', leave=False):
        images         = batch['images'].to(device)
        targets        = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Training-style forward gives us loss for free.
        loss, _ = model(images, targets=targets, target_lengths=target_lengths)
        total_loss += loss.item()

        # Autoregressive decode for sequence / CER metrics.
        # Use a conservative max_len cap during validation.
        max_len      = int(target_lengths.max().item()) * 2 + 5
        enc_feat     = model(images)   # inference mode → encoder features
        decoded      = model.greedy_autoregressive_decode(enc_feat, max_len=max_len)
        seq_acc, cer = compute_metrics(decoded, target_lengths, targets)

        total_seq += seq_acc
        total_cer += cer
        n_batches += 1

    avg_loss     = total_loss / max(1, n_batches)
    avg_seq_acc  = total_seq  / max(1, n_batches)
    avg_char_acc = 1.0 - total_cer / max(1, n_batches)
    return avg_loss, avg_seq_acc, avg_char_acc


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    optimizer: AdamW,
    lr_scheduler: WarmupCosineLR,
    device: torch.device,
    steps_per_epoch: int,
    clip_grad: float = 1.0,
    log_every: int = 50,
) -> tuple[float, float, float]:
    """Train for one epoch.

    Args:
        model:            The TransformerSeq2Seq model.
        dataloader:       Training DataLoader (rebuilt each epoch for curriculum).
        optimizer:        AdamW optimizer.
        lr_scheduler:     WarmupCosineLR scheduler (stepped per batch).
        device:           Training device.
        steps_per_epoch:  How many steps constitute one epoch.
        clip_grad:        Gradient clipping max-norm.
        log_every:        Log metrics every N steps.

    Returns:
        (avg_loss, avg_seq_acc, avg_char_acc)
    """
    model.train()
    total_loss = total_seq = total_cer = 0.0
    n_metric_steps = 0

    pbar = tqdm(enumerate(dataloader), total=steps_per_epoch,
                desc='  train', leave=False)

    for step, batch in pbar:
        if step >= steps_per_epoch:
            break

        images         = batch['images'].to(device)
        targets        = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        optimizer.zero_grad()

        # Training forward: (loss, logits).
        loss, logits = model(
            images,
            targets=targets,
            target_lengths=target_lengths,
        )
        loss.backward()

        # Gradient clipping — prevents exploding gradients in early training.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        # Step the warmup+cosine LR scheduler AFTER the optimiser step.
        # CRITICAL for Transformer stability.
        lr_scheduler.step()

        total_loss += loss.item()

        # Compute sequence-level metrics every log_every steps.
        if (step + 1) % log_every == 0 or (step + 1) == steps_per_epoch:
            with torch.no_grad():
                # Greedy decode using the training logits (cheapest option:
                # take argmax over the L positions, then strip special tokens).
                # Note: logits are from teacher-forcing, not autoregressive —
                # metrics here are "teacher-forced accuracy" (optimistic).
                preds = logits.argmax(dim=-1)   # [B, L+1]
                # Strip SOS/EOS/PAD from predicted tokens to get digit lists.
                decoded_fast: list[list[int]] = []
                for i, seq in enumerate(preds.tolist()):
                    L = int(target_lengths[i].item())
                    # Teacher-forced: positions 0..L correspond to digits 0..L-1
                    # and position L is the EOS slot.
                    digits = [t for t in seq[:L] if t < 10]
                    decoded_fast.append(digits)

                seq_acc, cer = compute_metrics(
                    decoded_fast, target_lengths, targets
                )

            total_seq      += seq_acc
            total_cer      += cer
            n_metric_steps += 1
            pbar.set_postfix(
                loss=f'{loss.item():.3f}',
                seq=f'{seq_acc:.3f}',
                cer=f'{cer:.3f}',
                lr=f'{lr_scheduler.current_lr:.2e}',
            )

    avg_loss     = total_loss / max(1, steps_per_epoch)
    avg_seq      = total_seq  / max(1, n_metric_steps)
    avg_char_acc = 1.0 - total_cer / max(1, n_metric_steps)
    return avg_loss, avg_seq, avg_char_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train TransformerSeq2Seq baseline for length-extrapolation ablation.')
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--max_length',      type=int,   default=12,
                        help='Max training sequence length — keep short for extrapolation test.')
    parser.add_argument('--checkpoint_dir',  type=str,
                        default='./checkpoints/transformer_baseline')
    parser.add_argument('--warmup_steps',    type=int,   default=2000,
                        help='CRITICAL for Transformer stability.')
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--clip_grad',       type=float, default=1.0)
    parser.add_argument('--num_workers',     type=int,   default=2)
    parser.add_argument('--train_size',      type=int,   default=100_000)
    parser.add_argument('--val_size',        type=int,   default=10_000)
    parser.add_argument('--early_stop',      type=int,   default=12)
    parser.add_argument('--resume',          type=str,   default='',
                        help='Path to existing checkpoint to resume from.')
    parser.add_argument('--d_model',         type=int,   default=256)
    parser.add_argument('--nhead',           type=int,   default=8)
    parser.add_argument('--num_layers',      type=int,   default=4)
    parser.add_argument('--dim_feedforward', type=int,   default=1024)
    parser.add_argument('--dropout',         type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_dir  = os.path.join(args.checkpoint_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    best_ckpt    = os.path.join(args.checkpoint_dir, 'best_transformer.pt')
    metrics_file = os.path.join(metrics_dir, 'transformer_metrics.csv')

    # ── Override shared config dict (same keys as CTC training) ───────────────
    # This is the exact same pattern used in train_colab_ctc_uncapped.ipynb.
    config['batch_size']        = args.batch_size
    config['max_seq_len_final'] = args.max_length   # cap training length
    config['train_size']        = args.train_size
    config['val_size']          = args.val_size
    config['num_workers']       = args.num_workers

    # ── Initialise CSV log ────────────────────────────────────────────────────
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_seq_acc', 'val_seq_acc',
                'train_char_acc', 'val_char_acc', 'lr',
            ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    # Reusing existing dataset/utils to maintain DRY principles and ensure
    # identical data pipeline for fair comparison.
    print('Building digit bank and validation loader...')
    digit_bank, val_loader, _ = get_dataloaders(data_path=config['data_path'])
    
    # ── CRITICAL FIX: Recreate val_loader with transformer_collate_fn ─────────
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=args.batch_size,
        collate_fn=transformer_collate_fn,  # <--- Ensures 2D targets for validation
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    
    steps_per_epoch = args.train_size // args.batch_size
    print(f'Steps per epoch : {steps_per_epoch}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TransformerSeq2Seq(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters : {n_params:,}')

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── LR Scheduler: warmup + cosine decay ───────────────────────────────────
    # CRITICAL for Transformer stability — without warmup, training diverges.
    total_steps  = args.epochs * steps_per_epoch
    lr_scheduler = WarmupCosineLR(
        optimizer    = optimizer,
        warmup_steps = args.warmup_steps,
        total_steps  = total_steps,
        base_lr      = args.lr,
        eta_min      = 1e-6,
    )

    # ── Optional resume ───────────────────────────────────────────────────────
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
        print(f'Resumed from epoch {ckpt["epoch"]} | val_seq_acc={best_val_seq:.4f}')

    early_stop_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        # Rebuild training dataset each epoch so augmentation curriculum updates.
        # Reusing InfiniteCTCDataset — identical to CTC baseline.
        train_ds = InfiniteCTCDataset(
            digit_bank, config,
            size=args.train_size,
            augment=True,
            epoch=epoch,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            collate_fn=transformer_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

        train_loss, train_seq, train_char = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler,
            device, steps_per_epoch,
            clip_grad=args.clip_grad,
        )
        val_loss, val_seq, val_char = validate(model, val_loader, device)

        current_lr = lr_scheduler.current_lr
        print(f'  Train | loss={train_loss:.4f}  seq_acc={train_seq:.4f}  '
              f'char_acc={train_char:.4f}')
        print(f'  Val   | loss={val_loss:.4f}  seq_acc={val_seq:.4f}  '
              f'char_acc={val_char:.4f}  lr={current_lr:.2e}')

        # Checkpoint on best val_seq_acc.
        if val_seq > best_val_seq:
            best_val_seq       = val_seq
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, val_seq, best_ckpt)
            print(f'  ✔ New best model saved (val_seq_acc={val_seq:.4f})')
        else:
            early_stop_counter += 1

        # CSV log.
        with open(metrics_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_seq, val_seq,
                train_char, val_char, current_lr,
            ])

        # Early stopping.
        if early_stop_counter >= args.early_stop:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

    print(f'\nTraining complete. Best val_seq_acc: {best_val_seq:.4f}')

    # Print training curve summary.
    print('\n── Training curve summary ───────────────────────────────────')
    print(f'  Metrics logged to : {metrics_file}')
    print(f'  Best checkpoint   : {best_ckpt}')
    print(f'  Best val_seq_acc  : {best_val_seq:.4f}')


if __name__ == '__main__':
    main()
