import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.transforms as transforms
from dataset import build_digit_bank, get_digit_transform, make_sequence

def main():
    print("Downloading MNIST and building digit bank...")
    # Using './data' relative to the execution directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    digit_bank = build_digit_bank(data_path, train=True)
    
    # Using augment=True to see the actual training distributions, or False for clean digits
    transform = get_digit_transform(augment=True)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'samples')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating 10 sample images...")
    for i in range(10):
        img_tensor, label = make_sequence(digit_bank, transform)
        
        # label contains [SOS, d1, d2, ..., EOS]
        digits = label[1:-1].tolist()
        digit_str = "".join(map(str, digits))
        
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(img_tensor)
        
        filename = os.path.join(out_dir, f"sample_{i+1}_{digit_str}.png")
        img.save(filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
