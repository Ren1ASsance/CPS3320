import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import AnimalCNN  # Ensure the path is correct

class AnimalEvaluator:
    def __init__(self, model_path, eval_dir):
        self.model_path = model_path
        self.eval_dir = eval_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        # Load evaluation dataset
        self.eval_data = datasets.ImageFolder(root=self.eval_dir, transform=self.transform)
        self.eval_loader = DataLoader(self.eval_data, batch_size=32, shuffle=False)
        self.num_classes = len(self.eval_data.classes)

        # Load trained model
        self.model = AnimalCNN(self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self):
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Per-class accuracy calculation
                for i in range(len(labels)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        # Print overall accuracy
        overall_acc = 100. * correct / total
        print(f"\nOverall Accuracy: {overall_acc:.2f}%")

        # Print per-class accuracy
        for i, class_name in enumerate(self.eval_data.classes):
            acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            print(f"{class_name}: {acc:.2f}%")
