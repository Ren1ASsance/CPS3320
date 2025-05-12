import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the CNN model for animal classification
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
        x = self.features(x)                      # Extract features
        x = x.view(x.size(0), -1)                 # Flatten
        x = self.classifier(x)                    # Classify
        return x

# Trainer class for training and validating the model
class AnimalTrainer:
    def __init__(self, train_dir, val_dir, checkpoint_dir='../checkpoints', num_epochs=30):
        self.train_dir = train_dir                      
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, '../animal_cnn.pth')
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Data augmentation and preprocessing for training
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        # Load training and validation datasets
        self.train_data = datasets.ImageFolder(root=self.train_dir, transform=self.transform)   

        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)

        self.num_classes = len(self.train_data.classes)
        print("Number of classes:", self.num_classes)
        print("Class to index mapping:", self.train_data.class_to_idx)

        self.model = AnimalCNN(self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    # Load model checkpoint if exists
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print("Model loaded from checkpoint.")

    # Evaluate model on validation set
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = 100. * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

    # Training loop
    def train(self):
        print("Using device:", self.device)
        self.load_checkpoint()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "accuracy": f"{100 * correct / total:.2f}"
                })

            # Print metrics at the end of each epoch
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{self.num_epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

            self.scheduler.step()
            torch.save(self.model.state_dict(), self.checkpoint_path)  # Save model after each epoch
            self.validate()

        return self.model
