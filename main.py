import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ==========================================
# 1. CUSTOM CNN ARCHITECTURE DEFINITION
# ==========================================
class CustomCNN(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) architecture designed for 
    image classification tasks (e.g., CIFAR-10: 32x32 images, 3 channels, 10 classes).
    """

    def __init__(self):
        super().__init__()

        # Feature extraction layers using convolutional blocks and max pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Classification layers mapping extracted spatial features to class probabilities
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # layer input dimensions: 64 channels * 8 * 8 spatial size]
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """
        Defines the forward pass data flow through feature extraction and classification blocks.
        """
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 2. TRAINING FUNCTION FOR A SINGLE EPOCH
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the neural network model for a single complete epoch over the training data loader.
    """
    model.train()  # Set model to training mode (enables dropout and batch normalization updates)
    total_loss = 0.0

    for images, labels in loader:
        # Transfer image batches and ground truth labels to the target computing device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward pass: compute model prediction logits
        outputs = model(images)

        # 2. Calculate the cross-entropy loss against true labels
        loss = criterion(outputs, labels)

        # 3. Backward pass: reset gradients, backpropagate error, and update optimizer weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return the average training loss across all batches
    return total_loss / len(loader)


# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate(model, loader, criterion, device):
    """
    Evaluates model performance metrics (average loss and classification accuracy) 
    over a given data loader without computing gradients.
    """
    model.eval()  # Set model to evaluation mode (disables training optimizations)

    total_loss = 0.0
    correct = 0
    total = 0

    # Execute evaluation block under inference mode to save memory and skip gradient tracking
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get predictions
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Find the class index with the highest logit score
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and percentage accuracy safely
    average_loss = total_loss / len(loader)
    accuracy = (100.0 * correct) / total

    return average_loss, accuracy


# ==========================================
# 4. MAIN SCRIPT EXECUTION PIPELINE
# ==========================================
def main():
    """
    Initializes datasets, applies data augmentations, configures data loaders, 
    executes multi-epoch training with validation tracking, and evaluates final performance.
    """

    # Select the optimal available computing device (CUDA GPU if available, else CPU)
    global train_accuracy, val_accuracy
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Define standard CIFAR-10 dataset channel normalization values (mean and standard deviation)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Apply data augmentation transforms for training images to improve generalization
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Use clean, consistent preprocessing without random augmentation for evaluation sets
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load the CIFAR-10 training dataset with random augmentations applied
    train_data = datasets.CIFAR10(
        root = "data",
        train = True,
        download = True,
        transform = train_transform,
    )

    # Load the exact same training dataset images without data augmentation for fair evaluation
    eval_data = datasets.CIFAR10(
        root = "data",
        train = True,
        download = False,
        transform = eval_transform,
    )

    # Split the training data into training (45,000 samples) and validation (5,000 samples) subsets using a fixed seed
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(50000, generator = generator)

    train_indices = indices[:45000].tolist()
    val_indices = indices[45000:].tolist()

    train_subset = Subset(train_data, train_indices)
    train_eval_subset = Subset(eval_data, train_indices)
    val_subset = Subset(eval_data, val_indices)

    # Load the separate CIFAR-10 test dataset for final unbiased performance evaluation
    test_data = datasets.CIFAR10(
        root = "data",
        train = False,
        download = True,
        transform = eval_transform,
    )

    # Create batch data loaders with requested formatting style
    train_loader = DataLoader(
        train_subset, batch_size = 64,  
        shuffle = True, num_workers = 0
    )

    train_eval_loader = DataLoader(
        train_eval_subset, batch_size = 64,  
        shuffle = False, num_workers = 0
    )

    val_loader = DataLoader(
        val_subset, batch_size = 64,  
        shuffle = False, num_workers = 0
    )

    test_loader = DataLoader(
        test_data, batch_size = 64,  
        shuffle = False, num_workers = 0
    )

    # Instantiate the custom CNN model and transfer it to the target device
    model = CustomCNN().to(device)

    # Define multi-class cross-entropy loss function and the Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr = 0.001
    )

    # Define training loop hyperparameters
    epochs = 5
    print("\n--- Starting Model Training & Validation ---")

    # Execute training and validation across configured epochs
    for epoch in range(epochs):

        # Train model for one epoch and capture average training loss
        train_loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device
        )

        # Evaluate model accuracy on unaugmented training subset
        _, train_accuracy = evaluate(
            model, train_eval_loader, criterion, device
        )

        # Evaluate model performance on validation subset
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        # Print telemetry stats for the completed epoch
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    # Evaluate the final trained model on the unseen test dataset
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    # Print summary performance comparison
    print("\n--- Final Accuracy Comparison ---")
    print(f"Training Accuracy:   {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Testing Accuracy:    {test_accuracy:.2f}%")
    print(f"Test Loss:           {test_loss:.4f}")

    # Save the trained model weights file to disk
    model_filename = "custom_cnn_augmented.pth"
    torch.save(
        model.state_dict(),
        model_filename
    )

    print(f"\nModel successfully saved to: {model_filename}")


# ==========================================
# 5. SCRIPT ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()