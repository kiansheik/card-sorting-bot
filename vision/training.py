import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

# Set the device to run on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define the image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Load the training and validation datasets
train_dataset = torchvision.datasets.ImageFolder(
    root="data/training",
    transform=transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
)

train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

validation_dataset = torchvision.datasets.ImageFolder(
    root="data/validation",
    transform=transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
)

validation_loader = data.DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(
            64 * IMAGE_SIZE * IMAGE_SIZE // (IMAGE_SIZE // 2) ** 2, num_classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Instantiate the model and move it to the device
model = CNN(num_classes=len(train_dataset.classes)).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Define the metrics to track
def accuracy(output, target, topk=(1,)):
    """Compute the top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Train the model
best_acc = 0.0
for epoch in range(10):  # Change this to a larger number of epochs to train for longer
    # Training
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss += loss.item()
        train_acc += accuracy(output, labels)[0].item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item()
            val_acc += accuracy(output, labels)[0].item()

    val_loss /= len(validation_loader)
    val_acc /= len(validation_loader)

    # Print the metrics
    print(
        f"Epoch {epoch+1}: train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%, val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%"
    )

    # Save the model if it's the best so far
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

print("Training complete!")
