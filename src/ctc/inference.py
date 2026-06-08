"""
CTC inference for the digit-sequence-reader.

Loads a trained CRNN_CTC checkpoint and predicts a digit string from a
single image (or a directory of images). Uses greedy decoding — for
production, swap in a beam search with a KenLM word/language model.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
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


def predict(image_path, checkpoint_path, visualize=False):
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

    if visualize:
        # Per-frame argmax over the time axis (i.e. the CNN downsampled width)
        frame_preds = logits.argmax(dim=-1)[0].cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                       gridspec_kw={'height_ratios': [1, 3]})
        ax1.imshow(orig_img, cmap='gray')
        ax1.axis('off')
        ax1.set_title(f'Predicted: {pred_string}')

        ax2.plot(frame_preds, marker='o', markersize=3, linewidth=1)
        ax2.set_ylim(-0.5, config['vocab_size'] - 0.5)
        ax2.set_xlabel('Encoder columns (CNN downsampled width)')
        ax2.set_ylabel('Argmax digit / blank')
        ax2.set_title('Per-frame predictions (10 = BLANK)')
        plt.tight_layout()
        plt.show()

    return pred_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--visualize',  action='store_true')
    args = parser.parse_args()

    pred = predict(args.image, args.checkpoint, args.visualize)
    print(f"Prediction: {pred}")


if __name__ == '__main__':
    main()
