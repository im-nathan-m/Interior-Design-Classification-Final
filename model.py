import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets

BATCH_SIZE = 100
NUM_EPOCHS = 50
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
ROOT_DIR = "archive"

# ------------ Step 0. GPU ------------
if torch.cuda.is_available():
	device = 'cuda'
	print('CUDA is available. Using GPU.')
else:
	device = 'cpu'

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

# ------------ Step 2. Data Info ------------
def data_info():
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
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1024 * 8 * 8, 19)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        X = self.pool(self.relu(self.conv1(X)))
        X = self.pool(self.relu(self.conv2(X)))
        X = self.pool(self.relu(self.conv3(X)))
        X = self.pool(self.relu(self.conv4(X)))
        X = self.relu(self.conv5(X))
        X = self.relu(self.conv6(X))
        X = X.flatten(start_dim=1)
        X = self.dropout(X)
        output = self.linear1(X)
        return output

# ------------ Step 4. Training & Validation Loop ------------
model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_correct = 0

    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        ### Get inputs and outputs in batches using the training DataLoader
        print(f"Batch {batch_idx}")

        train_x = train_x.to(device)
        train_y = train_y.to(device)

        train_preds = model(train_x)
        loss = criterion(train_preds, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        class_preds = train_preds.argmax(dim=1)  # returns the largest value in each tensor row
        train_correct += (class_preds == train_y).sum().item()

    train_accuracy = train_correct / len(train_dataset)

    ### Calculate the batch accuracy for training (see Week 5 Day 2 slides for reminder!)
    print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} Training | Loss: {loss.item()} | Accuracy: {train_accuracy} | Correct: {train_correct}")

    model.eval()
    val_correct = 0

    with torch.no_grad():
        for batch_idx, (val_x, val_y) in enumerate(val_loader):
            ### Get inputs and outputs in batches using the validation DataLoader
            print(f"Batch {batch_idx}")

            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_preds = model(val_x)
            loss = criterion(val_preds, val_y)

            class_preds = val_preds.argmax(dim=1)
            val_correct += (class_preds == val_y).sum().item()

        val_accuracy = val_correct / len(val_dataset)

        ### Calculate the batch accuracy for validation (see Week 5 Day 2 slides for reminder!)
        print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(val_loader)} Validation | Loss: {loss.item()} | Accuracy: {val_accuracy} | Correct: {val_correct}")

# ------------ Step 5. Testing Phase ------------
model.eval()
test_correct = 0

with torch.no_grad():
    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        ### Get inputs and outputs in batches using the test DataLoader
        print(f"Batch {batch_idx}")

        test_x = test_x.to(device)
        test_y = test_y.to(device)

        test_preds = model(test_x)
        loss = criterion(test_preds, test_y)

        class_preds = test_preds.argmax(dim=1)
        test_correct += (class_preds == test_y).sum().item()

    test_accuracy = test_correct / len(test_dataset)

    ### Calculate the batch accuracy for testing
    print(f"Testing | Loss: {loss.item()} | Accuracy: {test_accuracy} | Correct: {test_correct}")
