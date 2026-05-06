import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from config import config

def build_digit_bank(data_path, train=True):
    dataset = torchvision.datasets.MNIST(
        root=data_path, train=train, download=True
    )
    digit_bank = {i: [] for i in range(10)}
    for img, target in dataset:
        digit_bank[target].append(img)
    return digit_bank

def get_digit_transform(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(config['aug_rotation']),
            transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=config['aug_shear']),
            transforms.GaussianBlur(kernel_size=config['aug_blur_kernel'], sigma=config['aug_blur_sigma']),
            transforms.Resize((config['img_height'], config['img_height'])),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config['img_height'], config['img_height'])),
            transforms.ToTensor(),
        ])

def make_sequence(digit_bank, transform):
    seq_len = random.randint(config['min_seq_len'], config['max_seq_len'])
    digits = [random.randint(0, 9) for _ in range(seq_len)]
    
    img_tensors = []
    for d in digits:
        img = random.choice(digit_bank[d])
        img_tensor = transform(img)
        img_tensors.append(img_tensor)
        
    final_tensors = []
    for i, t in enumerate(img_tensors):
        final_tensors.append(t)
        if i < len(img_tensors) - 1:
            gap_width = random.randint(config['gap_min'], config['gap_max'])
            gap = torch.zeros(1, config['img_height'], gap_width)
            final_tensors.append(gap)
            
    final_img = torch.cat(final_tensors, dim=2)
    label = [config['SOS_IDX']] + digits + [config['EOS_IDX']]
    return final_img, torch.tensor(label, dtype=torch.long)

class SequenceDataset(Dataset):
    def __init__(self, data_path, size, augment=False, train=True):
        self.digit_bank = build_digit_bank(data_path, train=train)
        self.transform = get_digit_transform(augment)
        self.samples = []
        for _ in range(size):
            self.samples.append(make_sequence(self.digit_bank, self.transform))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    images, labels = zip(*batch)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    
    max_w = max(img.shape[2] for img in images)
    max_t = max(len(l) for l in labels)
    
    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=0)
        padded_images.append(padded_img)
        
    padded_labels = []
    for label in labels:
        pad_t = max_t - len(label)
        padded_label = torch.nn.functional.pad(label, (0, pad_t), value=config['PAD_IDX'])
        padded_labels.append(padded_label)
        
    return torch.stack(padded_images), torch.stack(padded_labels), lengths

def get_dataloaders(data_path=None):
    if data_path is None:
        data_path = config['data_path']
        
    train_ds = SequenceDataset(data_path, config['train_size'], augment=config['augment'], train=True)
    val_ds = SequenceDataset(data_path, config['val_size'], augment=False, train=False)
    test_ds = SequenceDataset(data_path, config['test_size'], augment=False, train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels, lengths = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")   # expect [64, 1, 64, W]
    print(f"Label batch shape : {labels.shape}")   # expect [64, T]
    print(f"First label       : {labels[0].tolist()}")
