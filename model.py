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

# ------------ Step 3. Model Class ------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(32 * 32 * 256, 19)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.pool(self.relu(self.conv1(X)))
        X = self.pool(self.relu(self.conv2(X)))
        X = self.relu(self.conv3(X))
        X = self.relu(self.conv4(X))
        X = X.flatten(start_dim=1)
        output = self.linear1(X)
        return output

model = ConvNet()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
NUM_EPOCHS = 20


# ------------ Step 4. Training Loop ------------
for epoch in range(NUM_EPOCHS):
    
    train_correct = 0

    for train_x, train_y in train_loader:
        ### Get inputs and outputs in batches using the training DataLoader
        train_preds = model(train_x)
        loss = criterion(train_preds, train_y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        class_preds = train_preds > 0
        train_correct += (class_preds.squeeze() == train_y).sum()
    
    train_accuracy = train_correct / len(train_dataset)

    ### Calculate the batch accuracy for training (see Week 5 Day 2 slides for reminder!)
    print(f"Epoch {epoch+1} Training | Loss: {loss.item()} | Accuracy: {train_accuracy} | Correct: {train_correct}")


    ### Include loop for validation dataset here.
    val_correct = 0

    for val_x, val_y in val_loader:
        ### Get inputs and outputs in batches using the validation DataLoader
        val_preds = model(val_x)
        loss = criterion(val_preds, val_y.unsqueeze(1))

        class_preds = val_preds > 0
        val_correct += (class_preds.squeeze() == val_y).sum()
    
    val_accuracy = val_correct / len(val_dataset)

    ### Calculate the batch accuracy for validation (see Week 5 Day 2 slides for reminder!)
    print(f"Epoch {epoch+1} Validation | Loss: {loss.item()} | Accuracy: {val_accuracy} | Correct: {val_correct}")
