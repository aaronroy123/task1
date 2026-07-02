import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure we don't need a results directory anymore since we save in root
# os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------
# 1. Architecture Definition
# ---------------------------------------------------------
class OverfittingMLP(nn.Module):
    """
    A highly complex Multi-Layer Perceptron designed to easily overfit 
    the MNIST dataset. It has multiple wide hidden layers.
    """
    def __init__(self):
        super(OverfittingMLP, self).__init__()
        # Flatten 28x28 image to 784 1D array
        self.flatten = nn.Flatten()
        
        # Extremely wide layers to memorize training data
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)  # 10 output classes for digits 0-9
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


class NormalMLP(nn.Module):
    """
    A standard, smaller Multi-Layer Perceptron designed to serve as a baseline.
    It has sufficient capacity to learn MNIST but is less prone to extreme overfitting.
    """
    def __init__(self):
        super(NormalMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# ---------------------------------------------------------
# 2. Data Preparation Function
# ---------------------------------------------------------
def get_dataloaders(use_augmentation=False, batch_size=128):
    """
    Loads MNIST and splits into 50k train, 10k val, 10k test.
    If use_augmentation is True, applies random rotations and shifts to train data.
    """
    
    # Base transform for validation and testing (just convert to tensor and normalize)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if use_augmentation:
        # Transform with data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = base_transform

    # Download datasets
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    val_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=base_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=base_transform
    )

    # Standard split: 60,000 total train data -> 50,000 train, 10,000 val
    torch.manual_seed(42)  # For reproducibility
    indices = torch.randperm(len(full_train_dataset)).tolist()
    train_indices = indices[:50000]
    val_indices = indices[50000:]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices) # Note: val uses base_transform

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# 3. Training Loop Function
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=20, weight_decay=0.0):
    """
    Trains the model and tracks training/validation loss and accuracy.
    weight_decay applies L2 regularization to the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    # We use Adam optimizer, applying weight_decay if provided
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # Logging
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # Load best model weights (Early Stopping simulation)
    model.load_state_dict(best_model_wts)
    return history


# ---------------------------------------------------------
# 4. Evaluation and Plotting
# ---------------------------------------------------------
def test_model(model, test_loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = correct / total
    print(f"\n{name} Test Accuracy: {acc * 100:.2f}%\n")
    return acc

def plot_history(history, title, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='red')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def plot_comparison(hist_dict, title, filename):
    plt.figure(figsize=(12, 5))
    
    # Plot Validation Loss
    plt.subplot(1, 2, 1)
    for label, hist in hist_dict.items():
        epochs = range(1, len(hist['val_loss']) + 1)
        plt.plot(epochs, hist['val_loss'], label=label)
    plt.title(f'{title} - Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    for label, hist in hist_dict.items():
        epochs = range(1, len(hist['val_acc']) + 1)
        plt.plot(epochs, hist['val_acc'], label=label)
    plt.title(f'{title} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Comparison plot saved to {filename}")
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    epochs = 20

    print("="*50)
    print("PHASE 0: BASELINE (NORMAL) MODEL")
    print("="*50)
    
    # 1. Data without augmentation
    train_loader_base, val_loader_base, test_loader = get_dataloaders(use_augmentation=False)
    
    # 2. Instantiate and train model
    model_normal = NormalMLP().to(device)
    print("Training Normal Baseline Model...")
    history_normal = train_model(
        model=model_normal, 
        train_loader=train_loader_base, 
        val_loader=val_loader_base, 
        epochs=epochs, 
        weight_decay=0.0
    )
    
    # 3. Test and plot
    test_model(model_normal, test_loader, name="Normal Model")
    plot_history(history_normal, "Phase 0: Baseline Normal Model", "phase_0_-_normal.png")
    print("\n")

    print("="*50)
    print("PHASE 1: OVERFITTING THE MODEL")
    print("="*50)
    
    # 1. Data without augmentation
    train_loader_base, val_loader_base, test_loader = get_dataloaders(use_augmentation=False)
    
    # 2. Instantiate and train model
    model_overfit = OverfittingMLP().to(device)
    print("Training Model without Regularization...")
    history_overfit = train_model(
        model=model_overfit, 
        train_loader=train_loader_base, 
        val_loader=val_loader_base, 
        epochs=epochs, 
        weight_decay=0.0 # No L2 regularization
    )
    
    # 3. Test and plot
    test_model(model_overfit, test_loader, name="Overfitted Model")
    plot_history(history_overfit, "Phase 1: Overfitting", "phase_1_-_overfitting.png")


    print("\n" + "="*50)
    print("PHASE 2: RESOLVING OVERFITTING (SAME ARCHITECTURE)")
    print("="*50)
    
    # 1. Data with augmentation
    train_loader_aug, val_loader_aug, _ = get_dataloaders(use_augmentation=True)
    
    # 2. Instantiate FRESH model (exact same architecture)
    model_regularized = OverfittingMLP().to(device)
    print("Training Model with L2 Regularization & Data Augmentation...")
    history_regularized = train_model(
        model=model_regularized, 
        train_loader=train_loader_aug, 
        val_loader=val_loader_aug, 
        epochs=epochs, 
        weight_decay=1e-4 # Add L2 Regularization
    )
    
    # 3. Test and plot
    test_model(model_regularized, test_loader, name="Regularized Model")
    plot_history(history_regularized, "Phase 2: Regularized", "phase_2_-_regularization.png")
    
    print("\n" + "="*50)
    print("PHASE 3: COMPARING ALL MODELS")
    print("="*50)
    
    histories = {
        "Baseline (Normal)": history_normal,
        "Overfitted": history_overfit,
        "Regularized": history_regularized
    }
    plot_comparison(histories, "Model Comparison", "comparison.png")
    
    print("\nProcess Complete!")
