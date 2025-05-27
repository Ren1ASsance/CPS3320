import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AnimalCNN

# Trainer class for training and validating the model
class AnimalTrainer:
    def __init__(self, train_dir, checkpoint_dir='../checkpoints', num_epochs=30,max_patience = 3):
        self.train_dir = train_dir                      
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, '../animal_cnn.pth')
        self.num_epochs = num_epochs
        self.max_patience = max_patience
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

        best_val_acc = 0.0  # Initialize best validation accuracy
        patience_counter = 0  # Counter to track how many epochs without improvement in validation accuracy

        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()  # Clear gradients from previous step
                outputs = self.model(inputs)  # Forward pass: get predictions from the model
                loss = self.criterion(outputs, labels)  # Compute loss

                loss.backward()  # Backward pass: compute gradients
                self.optimizer.step()  # Update weights using optimizer

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)  # Get the predicted class with the highest score
                correct += (predicted == labels).sum().item()  # Count correct predictions
                total += labels.size(0)

                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "accuracy": f"{100 * correct / total:.2f}"
                })

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{self.num_epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

            # Update learning rate scheduler
            self.scheduler.step()

            # Save model checkpoint after each epoch
            torch.save(self.model.state_dict(), self.checkpoint_path)

            # Validate the model after each epoch
            self.validate()

            # Check if validation accuracy improved
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc  # Update the best validation accuracy
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # If no improvement, increment patience counter

            # If patience counter exceeds max_patience, trigger early stopping
            if patience_counter >= self.max_patience:
                print(f"Early stopping triggered. Validation accuracy didn't improve for {self.max_patience} epochs.")
                break  # Stop training early

        return self.model

