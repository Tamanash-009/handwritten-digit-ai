# Handwritten Digit Recognizer using PyTorch
# (Rewritten from TensorFlow to PyTorch due to Python 3.14 incompatibility)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. Data Loading and Normalization
# ==========================================
# Transformation pipeline:
# - Convert image to PyTorch Tensor
# - Normalize with MNIST Native Mean (0.1307) and Std (0.3081)
#   (This scales pixels roughly between 0 and 1, centered at 0)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST dataset...")
# Download and load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Download and load test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 2. Build the Sequential Model
# ==========================================
# Using nn.Sequential to mimic the Keras Sequential API
model = nn.Sequential(
    nn.Flatten(),                   # Flatten 28x28 image to a 784 vector
    nn.Linear(28 * 28, 128),        # Hidden Dense Layer (128 neurons)
    nn.ReLU(),                      # Activation function
    nn.Linear(128, 10)              # Output Layer (10 classes for digits 0-9)
)

# ==========================================
# 3. Model Compilation (Loss & Optimizer)
# ==========================================
# CrossEntropyLoss combines Softmax and Negative Log Likelihood
criterion = nn.CrossEntropyLoss()
# Adam optimizer is a standard choice for training neural networks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. Training Loop
# ==========================================
print("\nStarting Training for 5 Epochs...")
device = torch.device("cpu") # Using CPU for simplicity/compatibility
model.to(device)

epochs = 5
for epoch in range(1, epochs + 1):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients, forward pass, backward pass, optimizer step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {epoch_acc:.2f}%')

print("Training Completed.")

# ==========================================
# 5. Visualization and Prediction
# ==========================================
def visualize_prediction():
    model.eval() # Set model to evaluation mode
    
    # Pick a random image from the test set
    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    
    # Run the model prediction
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    # Visualize
    # Undo normalization for display (approximate)
    image_display = image_tensor.numpy().squeeze() * 0.3081 + 0.1307
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image_display, cmap='gray')
    plt.title(f"True Label: {label}\nPredicted: {prediction}")
    plt.axis('off')
    
    print(f"\n[Visualization] Displaying image index {idx}")
    print(f"True Label: {label}")
    print(f"Model Prediction: {prediction}")
    print("Close the image window to finish script execution.")
    
    plt.show()

if __name__ == "__main__":
    visualize_prediction()
