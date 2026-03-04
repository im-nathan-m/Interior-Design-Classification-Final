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

# ------------ Step 1. Data Preprocessing ------------
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

# ------------ Step 2. Data Info ------------
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
    # test & val aren't shuffled so class labels are all 0
    for inputs, targets in loader:
        print("INPUTS")
        print(inputs)
        print("CLASS LABELS (TARGETS)")
        print(targets)
        break

# ------------ Step 3. Model Init ------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(32 * 32 * 256, 19)
        self.relu = nn.Relu()

    def forward(self, X):
        X = self.pool(self.relu(self.conv1(X)))
        X = self.pool(self.relu(self.conv2(X)))
        X = self.relu(self.conv3(X))
        X = self.relu(self.conv4(X))
        X = X.flatten(start_dim=1)
        output = self.fc1(X)
        return output
