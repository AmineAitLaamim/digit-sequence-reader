import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import config
from .model import Seq2Seq

def compute_accuracy(logits, aligned_targets, lengths):
    """Compute sequence- and character-level accuracy.

    Args:
        logits:           [B, T, vocab_size]  — model output (length T = target_len-1)
        aligned_targets:  [B, T]              — targets[:, 1:] (SOS stripped, padded with PAD_IDX)
        lengths:          [B]                 — ORIGINAL full length including SOS
                                                   (length of unshifted `targets`)

    The mask MUST be built from `aligned_targets` (not from the original `targets`),
    otherwise the SOS slot — which exists in `targets` but not in `aligned_targets` —
    is counted as a valid token, producing `char_acc > 1.0`. The original-length
    `lengths` is used only to truncate past the true EOS (defence-in-depth: in case
    a sequence's `aligned_targets` somehow contains a non-PAD token after EOS).
    """
    preds = logits.argmax(dim=-1)

    seq_acc_count = 0
    char_correct = 0
    char_total = 0

    B = aligned_targets.size(0)
    for i in range(B):
        # Mask is over aligned_targets, not the original targets — this is the fix.
        mask = aligned_targets[i] != config['PAD_IDX']
        # Truncate to the real (length-1) valid positions (lengths includes SOS).
        real_len = max(0, int(lengths[i].item()) - 1)
        if real_len < mask.size(0):
            mask = mask.clone()
            mask[real_len:] = False

        valid_preds = preds[i][mask]
        valid_targets = aligned_targets[i][mask]

        if valid_targets.numel() > 0 and torch.equal(valid_preds, valid_targets):
            seq_acc_count += 1

        char_correct += (valid_preds == valid_targets).sum().item()
        char_total += valid_targets.numel()

    return seq_acc_count / B, char_correct / max(1, char_total)

def get_tf_ratio(epoch):
    """Linear scheduled sampling: anneal TF ratio tf_start -> tf_end over tf_anneal_epochs."""
    t0 = config['tf_start']
    t1 = config['tf_end']
    span = max(1, config['tf_anneal_epochs'])
    frac = min(1.0, max(0.0, (epoch - 1) / span))
    return t0 + (t1 - t0) * frac

def train_epoch(model, dataloader, optimizer, criterion, device, steps_per_epoch, tf_ratio):
    model.train()
    total_loss = 0
    total_seq_acc = 0
    total_char_acc = 0

    pbar = tqdm(enumerate(dataloader), total=steps_per_epoch, desc="Training", leave=False)
    for i, (images, targets, lengths) in pbar:
        if i >= steps_per_epoch:
            break

        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        logits, alphas = model(images, targets, teacher_forcing_ratio=tf_ratio)
        
        # Align targets for loss: we predict targets from index 1 to end
        aligned_targets = targets[:, 1:]
        
        # Flatten for CrossEntropy
        logits_flat = logits.reshape(-1, config['vocab_size'])
        targets_flat = aligned_targets.reshape(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        optimizer.step()
        
        total_loss += loss.item()
        
        seq_acc, char_acc = compute_accuracy(logits, aligned_targets, lengths)
        total_seq_acc += seq_acc
        total_char_acc += char_acc
        
    return total_loss / steps_per_epoch, total_seq_acc / steps_per_epoch, total_char_acc / steps_per_epoch

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_seq_acc = 0
    total_char_acc = 0
    
    with torch.no_grad():
        for images, targets, lengths in tqdm(dataloader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            
            # teacher_forcing_ratio=0.0 during eval
            logits, alphas = model(images, targets, teacher_forcing_ratio=0.0)
            
            aligned_targets = targets[:, 1:]
            
            logits_flat = logits.reshape(-1, config['vocab_size'])
            targets_flat = aligned_targets.reshape(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            total_loss += loss.item()
            
            seq_acc, char_acc = compute_accuracy(logits, aligned_targets, lengths)
            total_seq_acc += seq_acc
            total_char_acc += char_acc
            
    n = len(dataloader)
    return total_loss / n, total_seq_acc / n, total_char_acc / n

def save_checkpoint(model, optimizer, epoch, val_loss, val_seq_acc, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_seq_acc': val_seq_acc,
        'config': config
    }
    torch.save(checkpoint, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_path', type=str, required=True, help='Path to Google Drive base folder')
    parser.add_argument('--epochs', type=int, default=config['epochs'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--resume', type=str, default='', help='Path to best_model.pt to resume')
    args = parser.parse_args()
    
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['drive_path'] = args.drive_path
    
    checkpoint_dir = os.path.join(args.drive_path, 'checkpoints')
    metrics_dir = os.path.join(args.drive_path, 'metrics')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_seq_acc', 'val_seq_acc', 'train_char_acc', 'val_char_acc', 'lr'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from .dataset_aggressive import get_dataloaders, InfiniteSequenceDataset, collate_fn as agg_collate_fn
    from torch.utils.data import DataLoader
    
    multidigit_bank, val_loader, test_loader = get_dataloaders(data_path=config['data_path'])
    steps_per_epoch = config['train_size'] // config['batch_size']
    
    model = Seq2Seq().to(device)
    optimizer = Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=config['PAD_IDX'])
    # Schedule on val_seq_acc (mode='max') so the LR only drops when the
    # actual metric stalls — not when a noisy val_loss spike is purely
    # an exposure-bias artifact from free-running inference.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=config['lr_patience'], factor=config['lr_factor'], min_lr=config['lr_min'])

    start_epoch = 1
    best_val_seq = 0.0
    best_val_loss = float('inf')

    # resume always points to best_model.pt
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_seq = ckpt.get('val_seq_acc', 0.0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {ckpt['epoch']} | val_seq_acc={best_val_seq:.4f}")

    early_stop_counter = 0

    for epoch in range(start_epoch, config['epochs'] + 1):
        tf_ratio = get_tf_ratio(epoch)
        print(f"\nEpoch {epoch}/{config['epochs']} | tf_ratio={tf_ratio:.3f}")

        # Recreate train_loader each epoch to update the augmentation intensity curriculum
        train_ds = InfiniteSequenceDataset(multidigit_bank, config, size=config['train_size'], augment=True, epoch=epoch)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], collate_fn=agg_collate_fn, num_workers=config.get('num_workers', 4), pin_memory=True, persistent_workers=True)

        train_loss, train_seq, train_char = train_epoch(
            model, train_loader, optimizer, criterion, device, steps_per_epoch, tf_ratio
        )
        val_loss, val_seq, val_char = val_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train | loss={train_loss:.4f} seq_acc={train_seq:.4f} char_acc={train_char:.4f}")
        print(f"Val   | loss={val_loss:.4f} seq_acc={val_seq:.4f} char_acc={val_char:.4f} | lr={current_lr}")

        # ── Divergence guard ──────────────────────────────────────────────
        # If val_loss explodes beyond div_guard_mult × best loss, the model
        # has likely collapsed. Roll weights back to the best checkpoint
        # so subsequent epochs start from a sane state.
        if best_val_loss < float('inf') and val_loss > config['div_guard_mult'] * best_val_loss:
            print(f"  ⚠ Divergence detected (val_loss={val_loss:.4f} > {config['div_guard_mult']}×best={best_val_loss:.4f}). Rolling back to best checkpoint.")
            best_ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
            if os.path.exists(best_ckpt_path):
                rollback = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(rollback['model_state_dict'], strict=False)
                optimizer.load_state_dict(rollback['optimizer_state_dict'])
            val_loss = best_val_loss
            val_seq = best_val_seq

        scheduler.step(val_seq)

        if val_seq > best_val_seq:
            best_val_seq = val_seq
            best_val_loss = val_loss
            early_stop_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_seq,
                path=os.path.join(checkpoint_dir, 'best_model.pt')
            )
            print(f"  ✓ New best model saved (val_seq_acc={val_seq:.4f})")
        else:
            early_stop_counter += 1

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_seq, val_seq, train_char, val_char, current_lr])

        if early_stop_counter >= config['early_stop_patience']:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

if __name__ == '__main__':
    main()
