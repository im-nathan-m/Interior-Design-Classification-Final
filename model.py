import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BATCH_SIZE = 100
ROOT_DIR = "archive"

# ------------ Step 1. Data Visualization ------------
pass

# ------------ Step 2. Data Preprocessing ------------
train_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_val', transform=test_transforms)
test_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/dataset_test', transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(train_dataset.classes)

NUM_CLASSES = len(train_dataset.classes)