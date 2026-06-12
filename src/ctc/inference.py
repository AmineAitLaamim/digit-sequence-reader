"""
CTC inference for the digit-sequence-reader.

Loads a trained CRNN_CTC checkpoint and predicts a digit string from a
single image. Uses greedy decoding — for production, swap in a beam search
with a KenLM word/language model.

Accuracy metrics (optional):
    If `--ground-truth "12345"` is passed (or can be auto-extracted from
    the image filename, e.g. `sample_L7_1234567.png` → "1234567"), the
    script computes both Character Error Rate (CER) and sequence
    accuracy and prints them in the FINAL RESULT banner.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import re
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from .config import config
from .model import CRNN_CTC, greedy_decode


def preprocess_image(image_path):
    """Load a grayscale digit-sequence image and convert it to a tensor."""
    img = Image.open(image_path).convert('L')
    w, h = img.size
    new_w = int(w * (config['img_height'] / h))

    # Snap width up to a multiple of 8 (the CNN downsamples by 8x).
    if new_w % 8 != 0:
        new_w = ((new_w + 7) // 8) * 8

    transform = transforms.Compose([
        transforms.Resize((config['img_height'], new_w)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)   # [1, 1, 64, W]
    return img_tensor, img


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


def auto_extract_gt(image_path):
    """
    Try to pull the ground-truth digit string out of the image filename.
    Looks for any contiguous run of digits at the END of the basename,
    ignoring the file extension.

    Examples:
        sample_L7_1234567.png      -> '1234567'
        /path/to/test_42.png      -> '42'
        ctc_sample_1_98765.png    -> '98765'
        photo.png                 -> None
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    # Find the last run of digits (one or more) at the end of the name.
    m = re.search(r'(\d+)$', basename)
    return m.group(1) if m else None


def predict(image_path, checkpoint_path, visualize=False, ground_truth=None):
    """Run inference and return (pred_string, metrics_dict_or_None)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN_CTC().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # strict=False allows loading checkpoints that pre-date a refactor
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    img_tensor, orig_img = preprocess_image(image_path)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))    # [1, T, V]
    decoded = greedy_decode(logits)[0]
    pred_string = "".join(str(d) for d in decoded)

    metrics = None
    if ground_truth:
        gt_list = [int(c) for c in ground_truth if c.isdigit()]
        pred_list = [int(c) for c in pred_string if c.isdigit()]

        # Sequence accuracy: exact match.
        seq_match = (gt_list == pred_list)
        # CER via Levenshtein edit distance.
        edit_dist = _levenshtein(pred_list, gt_list)
        cer = edit_dist / max(1, len(gt_list))
        metrics = {
            'ground_truth':     ground_truth,
            'pred':             pred_string,
            'seq_match':        seq_match,
            'cer':              cer,
            'edit_distance':    edit_dist,
            'gt_length':        len(gt_list),
            'pred_length':      len(pred_list),
        }

    if visualize:
        # Per-frame argmax over the time axis (i.e. the CNN downsampled width)
        frame_preds = logits.argmax(dim=-1)[0].cpu().numpy()
        title_suffix = ''
        if metrics is not None:
            tick = 'OK' if metrics['seq_match'] else 'X'
            title_suffix = f"  [GT={metrics['ground_truth']}  match={tick}  CER={metrics['cer']:.2f}]"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                       gridspec_kw={'height_ratios': [1, 3]})
        ax1.imshow(orig_img, cmap='gray')
        ax1.axis('off')
        ax1.set_title(f'Predicted: {pred_string}{title_suffix}')

        ax2.plot(frame_preds, marker='o', markersize=3, linewidth=1)
        ax2.set_ylim(-0.5, config['vocab_size'] - 0.5)
        ax2.set_xlabel('Encoder columns (CNN downsampled width)')
        ax2.set_ylabel('Argmax digit / blank')
        ax2.set_title('Per-frame predictions (10 = BLANK)')
        plt.tight_layout()
        plt.show()

    return pred_string, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',         type=str, required=True)
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--visualize',     action='store_true')
    parser.add_argument('--ground-truth',  type=str, default=None,
                        help='Optional ground-truth digit string '
                             '(e.g. "12345"). If omitted, the script will '
                             'try to auto-extract digits from the image '
                             'filename (e.g. sample_L7_1234567.png).')
    parser.add_argument('--no-gt',         action='store_true',
                        help='Skip ground-truth comparison even if the '
                             'filename contains digits.')
    args = parser.parse_args()

    # Determine ground truth
    gt = None
    if not args.no_gt:
        gt = args.ground_truth or auto_extract_gt(args.image)

    pred, metrics = predict(args.image, args.checkpoint, args.visualize,
                            ground_truth=gt)

    # Loud, unambiguous final-result banner.
    print()
    print("=" * 60)
    print(f"  FINAL RESULT:  {pred}")
    print(f"  (length: {len(pred)} digits)")

    if metrics is not None:
        tick = 'OK ✔ (exact match)' if metrics['seq_match'] else 'X ✘ (mismatch)'
        print("-" * 60)
        print(f"  Ground truth  :  {metrics['ground_truth']}")
        print(f"  Sequence acc  :  {tick}")
        print(f"  Edit distance :  {metrics['edit_distance']}")
        print(f"  CER           :  {metrics['cer']:.4f}  "
              f"({metrics['edit_distance']} / {metrics['gt_length']})")
    elif gt is None:
        print("-" * 60)
        print("  (No ground truth available — pass --ground-truth \"12345\"")
        print("   to compute sequence accuracy and CER.)")
    print("=" * 60)


if __name__ == '__main__':
    main()
