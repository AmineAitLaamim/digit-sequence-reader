import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import config
from dataset_aggressive import get_dataloaders
from model import Seq2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--drive_path', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, test_loader = get_dataloaders()
    
    model = Seq2Seq().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()
    all_targets_cm = []
    all_preds_cm = []
    
    seq_acc_count = 0
    char_correct = 0
    char_total = 0
    
    samples_for_attention = []
    
    with torch.no_grad():
        for images, targets, lengths in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits, alphas = model(images, targets=None, teacher_forcing_ratio=0.0)
            preds = logits.argmax(dim=-1)
            
            for i in range(targets.size(0)):
                if len(samples_for_attention) < 3:
                    samples_for_attention.append({
                        'image': images[i].cpu(),
                        'target': targets[i].cpu(),
                        'pred': preds[i].cpu(),
                        'alpha': alphas[i].cpu()
                    })
                    
                mask = targets[i] != config['PAD_IDX']
                valid_targets = targets[i][mask]
                actual_target = valid_targets[1:] # remove SOS
                
                pred_seq = []
                for p in preds[i]:
                    pred_seq.append(p.item())
                    if p.item() == config['EOS_IDX']:
                        break
                pred_seq_tensor = torch.tensor(pred_seq, dtype=torch.long)
                
                if torch.equal(pred_seq_tensor, actual_target.cpu()):
                    seq_acc_count += 1
                    
                min_len = min(len(pred_seq_tensor), len(actual_target))
                char_correct += (pred_seq_tensor[:min_len] == actual_target.cpu()[:min_len]).sum().item()
                char_total += len(actual_target)
                
                for pt, tt in zip(pred_seq_tensor[:min_len], actual_target.cpu()[:min_len]):
                    if tt.item() < 10 and pt.item() < 10:
                        all_targets_cm.append(tt.item())
                        all_preds_cm.append(pt.item())

    seq_acc = seq_acc_count / len(test_loader.dataset)
    char_acc = char_correct / max(1, char_total)
    
    metrics = {
        "sequence_accuracy": seq_acc,
        "character_accuracy": char_acc,
        "total_samples": len(test_loader.dataset),
        "checkpoint": os.path.basename(args.checkpoint)
    }
    
    metrics_path = os.path.join(args.drive_path, 'metrics')
    os.makedirs(metrics_path, exist_ok=True)
    
    with open(os.path.join(metrics_path, 'metrics_test.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    cm = confusion_matrix(all_targets_cm, all_preds_cm, labels=list(range(10)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Digit Confusion Matrix')
    plt.savefig(os.path.join(metrics_path, 'confusion_matrix.png'))
    plt.close()
    
    for idx, sample in enumerate(samples_for_attention):
        img = sample['image'][0].numpy()
        alpha = sample['alpha'].numpy()
        pred = sample['pred'].numpy()
        
        pred_len = 0
        for p in pred:
            pred_len += 1
            if p == config['EOS_IDX']:
                break
        alpha = alpha[:pred_len, :]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
        
        ax1.imshow(img, cmap='gray')
        ax1.axis('off')
        ax1.set_title(f'Predicted: {pred[:pred_len]}')
        
        ax2.imshow(alpha, cmap='hot', aspect='auto')
        ax2.set_xlabel('Encoder Columns (Image Space)')
        ax2.set_ylabel('Decoding Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_path, f'attention_sample_{idx}.png'))
        plt.close()

if __name__ == '__main__':
    main()
