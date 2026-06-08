import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ssl

# Bypass SSL verification for torchvision downloads (USPS specifically)
ssl._create_default_https_context = ssl._create_unverified_context

from config import config

def build_multidigit_bank(data_path='./data'):
    os.makedirs(data_path, exist_ok=True)
    bank = {i: [] for i in range(10)}
    
    # EMNIST Digits
    print("Loading EMNIST Digits...")
    emnist = datasets.EMNIST(root=data_path, split='digits', train=True, download=True)
    for img, label in emnist:
        bank[label].append(img)
        
    # QMNIST
    print("Loading QMNIST...")
    qmnist = datasets.QMNIST(root=data_path, what='train', compat=True, download=True)
    for img, label in qmnist:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        bank[label].append(img)
        
    # USPS
    print("Loading USPS...")
    usps = datasets.USPS(root=data_path, train=True, download=True)
    for img, label in usps:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = img.resize((28, 28), Image.BILINEAR)
        bank[label].append(img)
        
    for i in range(10):
        print(f"Digit {i}: {len(bank[i])} images")
        
    return bank

def get_dataloaders(data_path=None):
    from torch.utils.data import DataLoader
    
    data_path = data_path or config['data_path']
    multidigit_bank = build_multidigit_bank(data_path)

    # val and test use clean pipeline, no overlap, fixed sizes
    val_ds   = InfiniteSequenceDataset(multidigit_bank, config, size=config['val_size'], augment=False)
    test_ds  = InfiniteSequenceDataset(multidigit_bank, config, size=config['test_size'], augment=False)
    
    num_workers = config.get('num_workers', 4)
    val_loader  = DataLoader(val_ds, batch_size=config['batch_size'], collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    # train loader is recreated in train.py each epoch, so we only return the bank and val/test loaders
    return multidigit_bank, val_loader, test_loader

def get_digit_aug_pipeline(augment=True, config=None, epoch=1):
    if config is None:
        from config import config
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if not augment:
        # Clean pipeline for validation/test
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    intensity = min(1.0, epoch / config.get('aug_warmup_epochs', 10))

    noise_var = config.get('aug_noise_var', (5, 20))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(A, 'GaussianNoise'):
            noise = A.GaussianNoise(std_range=tuple(v * intensity for v in noise_var))
        else:
            try:
                noise = A.GaussNoise(std_range=tuple(v * intensity for v in noise_var))
            except Exception:
                noise = A.GaussNoise(var_limit=(noise_var[0]**2 * intensity, noise_var[1]**2 * intensity))

    transform = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=config.get('aug_brightness', 0.3) * intensity,
            contrast_limit=config.get('aug_contrast', 0.3) * intensity,
            p=0.5
        ),
        A.OneOf([
            noise,
            A.MotionBlur(blur_limit=max(3, int(config.get('aug_blur_limit', 3) * intensity)))
        ], p=0.4 * intensity),
        A.CoarseDropout(
            max_holes=8,
            max_height=8,
            max_width=8,
            p=config.get('aug_erasing_p', 0.3) * intensity
        ),
        A.Resize(64, 64),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])

    def apply(pil_image):
        np_img = np.array(pil_image, dtype=np.uint8)
        return transform(image=np_img)['image']

    return apply


def get_curriculum_max_len(epoch, config):
    """Linearly grow max sequence length from max_seq_len to max_seq_len_final
    after aug_warmup_epochs, over the next 20 epochs."""
    base   = config['max_seq_len']                         # 7
    final  = config.get('max_seq_len_final', 12)           # 12
    warmup = config.get('aug_warmup_epochs', 10)           # 10
    if epoch <= warmup:
        return base
    progress = min(1.0, (epoch - warmup) / 20.0)
    return int(base + progress * (final - base))


def make_sequence(digit_bank, aug_pipeline, config, augment=False, epoch=1):
    max_len = get_curriculum_max_len(epoch, config) if augment else config['max_seq_len']
    L = random.randint(config['min_seq_len'], max_len)
    digits = [random.randint(0, 9) for _ in range(L)]
    
    # Augmentation intensity — controls brightness, noise, dropout
    aug_intensity = min(1.0, epoch / config.get('aug_warmup_epochs', 10))
    
    # Overlap probability — completely independent from augmentation intensity
    if augment and epoch > config.get('overlap_start_epoch', 5):
        overlap_intensity = min(1.0, (epoch - config.get('overlap_start_epoch', 5)) / 10.0)
        overlap_prob = config.get('overlap_prob_max', 0.10) * overlap_intensity
    else:
        overlap_prob = 0.0
    
    sequence_parts = []
    for i, digit in enumerate(digits):
        img_pil = random.choice(digit_bank[digit])
        img_tensor = aug_pipeline(img_pil)  # [1, 64, 64]
        
        if i > 0:
            if augment and random.random() < overlap_prob:
                gap = -random.randint(1, config.get('overlap_max', 8))
            else:
                gap = random.randint(config.get('gap_min', 0), config.get('gap_max', 12))
                
            if gap >= 0:
                spacer = torch.zeros(1, 64, gap)
                sequence_parts.append(spacer)
            elif gap < 0:
                # trim the right side of the last item in sequence_parts
                if sequence_parts:
                    sequence_parts[-1] = sequence_parts[-1][:, :, :-abs(gap)]
                    
        sequence_parts.append(img_tensor)
        
    seq_img = torch.cat(sequence_parts, dim=2)  # [1, 64, W]
    
    label = [config['SOS_IDX']] + digits + [config['EOS_IDX']]
    label_tensor = torch.tensor(label, dtype=torch.long)
    
    return seq_img, label_tensor

class InfiniteSequenceDataset(IterableDataset):
    def __init__(self, digit_bank, config, size=None, augment=True, epoch=1):
        super().__init__()
        self.digit_bank = digit_bank
        self.config = config
        self.size = size
        self.augment = augment
        self.epoch = epoch
        self.aug_pipeline = get_digit_aug_pipeline(augment=self.augment, config=self.config, epoch=self.epoch)
        
    def __len__(self):
        if self.size is None:
            raise TypeError("Dataset has no fixed size.")
        return self.size
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_size = self.size
        
        if worker_info is not None:
            np.random.seed(np.random.get_state()[1][0] + worker_info.id)
            random.seed(random.getstate()[1][0] + worker_info.id)
            
            if self.size is not None:
                worker_size = self.size // worker_info.num_workers
                # handle remainder for the last worker
                if worker_info.id == worker_info.num_workers - 1:
                    worker_size += self.size % worker_info.num_workers
            
        count = 0
        while True:
            if worker_size is not None and count >= worker_size:
                break
            yield make_sequence(self.digit_bank, self.aug_pipeline, self.config, augment=self.augment, epoch=self.epoch)
            count += 1

def collate_fn(batch):
    images, labels = zip(*batch)
    
    lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    max_len = lengths.max().item()
    
    max_w = max([img.size(2) for img in images])
    
    B = len(batch)
    padded_images = torch.zeros(B, 1, 64, max_w)
    padded_labels = torch.full((B, max_len), config['PAD_IDX'], dtype=torch.long)
    
    for i in range(B):
        w = images[i].size(2)
        padded_images[i, :, :, :w] = images[i]
        
        l = len(labels[i])
        padded_labels[i, :l] = labels[i]
        
    return padded_images, padded_labels, lengths
