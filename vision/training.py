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
    root="path/to/train/dataset",
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
    root="path/to/validation/dataset",
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
