import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from config import config
from model import Seq2Seq

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    
    w, h = img.size
    new_w = int(w * (config['img_height'] / h))
    
    transform = transforms.Compose([
        transforms.Resize((config['img_height'], new_w)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def greedy_decode(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        images = image_tensor.to(device)
        logits, alphas = model(images, targets=None, teacher_forcing_ratio=0.0)

        preds = logits.argmax(dim=-1)[0].cpu().numpy()
        alpha = alphas[0].cpu().numpy()

        pred_digits = []
        pred_len = 0
        for p in preds:
            pred_len += 1
            if p == config['EOS_IDX']:
                break
            if p < 10:
                pred_digits.append(str(p))

        pred_string = "".join(pred_digits)
        return pred_string, alpha[:pred_len, :]

def predict(image_path, checkpoint_path, visualize=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # strict=False allows loading checkpoints that pre-date the pos_proj layer
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    img_tensor, orig_img = preprocess_image(image_path)
    pred_string, alpha = greedy_decode(model, img_tensor, device)
    
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
        ax1.imshow(orig_img, cmap='gray')
        ax1.axis('off')
        ax1.set_title(f'Predicted: {pred_string}')
        
        ax2.imshow(alpha, cmap='hot', aspect='auto')
        ax2.set_xlabel('Encoder Columns (Image Space)')
        ax2.set_ylabel('Decoding Steps')
        
        plt.tight_layout()
        plt.show()
        
    return pred_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    pred = predict(args.image, args.checkpoint, args.visualize)
    print(f"Prediction: {pred}")

if __name__ == '__main__':
    main()
