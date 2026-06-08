import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.transforms as transforms
from .dataset_aggressive import build_multidigit_bank, get_digit_aug_pipeline, make_sequence
from .config import config

def main():
    print("Downloading EMNIST, QMNIST, USPS and building digit bank...")
    # Using './data' relative to the execution directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    digit_bank = build_multidigit_bank(data_path)
    
    # Using augment=True at max epoch to see the actual training distribution
    epoch = config.get('epochs', 30)
    aug_pipeline = get_digit_aug_pipeline(augment=True, config=config, epoch=epoch)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'samples')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating 20 sample images at curriculum epoch {epoch}...")
    for i in range(20):
        img_tensor, label_tensor = make_sequence(digit_bank, aug_pipeline, config, augment=True, epoch=epoch)
        
        # label_tensor contains [SOS, d1, d2, ..., EOS]
        digits = label_tensor[1:-1].tolist()
        digit_str = "".join(map(str, digits))
        
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(img_tensor)
        
        filename = os.path.join(out_dir, f"sample_{i+1}_{digit_str}.png")
        img.save(filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
