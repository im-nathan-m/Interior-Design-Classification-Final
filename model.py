import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets, models

BATCH_SIZE = 100
ROOT_DIR = "archive"

# ------------ Step 1. Data Visualization ------------
pass

# ------------ Step 2. Data Preprocessing ------------
train_transforms = v2.Compose([
    v2.Resize((128, 128)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(5),
    v2.RandomAffine(
    degrees=0,
    translate=(0.05, 0.05),
    scale=(0.95, 1.05),
    ),
    v2.ColorJitter(
    brightness=0.3,
    contrast=0.3,
    saturation=0.2,
    hue=0.03
    ),
    v2.ToTensor()
])

test_transforms = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToTensor()
])

train_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_val', transform=test_transforms)
test_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_test', transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------ Step 3. Data Info ------------

for i, class_name in enumerate(train_dataset.classes, start=1):
    print(f"{i}. {class_name}")

print("Train/Val/Test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))

loaders = {
    "TRAIN": train_loader,
    "VAL": val_loader,
    "TEST": test_loader
}

for name, loader in loaders.items():
    print(f"\n{name} BATCH")
    
    # prints one batch per loader
    for inputs, targets in loader:
        print(inputs)
        print(targets)
        break