import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
class AnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalCNN, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),       # 3 input channels (RGB), 32 output channels, 3x3 kernel
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # Downsample by 2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),          # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)           # Output layer with `num_classes` units
        )

    def forward(self, x):
        x = self.features(x)                       # Extract features
        x = x.view(x.size(0), -1)                 # Flatten
        x = self.classifier(x)                    # Classify
        return x