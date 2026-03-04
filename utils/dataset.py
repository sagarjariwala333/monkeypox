"""
utils/dataset.py — MSLD v2.0 Dataset loader + augmentation pipeline
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    targets = np.array(dataset.targets)
    counts  = np.bincount(targets, minlength=len(dataset.classes))
    weights = 1.0 / (counts + 1e-6)
    return torch.FloatTensor(weights / weights.sum() * len(dataset.classes))

def get_kfold_loaders(data_dir, n_splits=5, batch_size=16, num_workers=2, seed=42):
    base = ImageFolder(root=data_dir)
    targets = np.array(base.targets)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out  = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        tr_ds = Subset(ImageFolder(root=data_dir, transform=get_train_transform()), tr_idx)
        va_ds = Subset(ImageFolder(root=data_dir, transform=get_val_transform()),   va_idx)
        out.append((
            DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True),
            DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        ))
    return out, base
