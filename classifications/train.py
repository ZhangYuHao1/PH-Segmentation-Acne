import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

# Define the transformation for data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB (EfficientNet expects 3 channels)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Load your dataset using torchvision.datasets.ImageFolder
train_dataset = datasets.ImageFolder(root='***', transform=transform)
test_dataset = datasets.ImageFolder(root='***', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the pre-trained EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)  # Adjust the number of classes

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate

# Initialize variables to track the best model
best_accuracy = 0.0
best_model_state_dict = None

# Training loop
num_epochs = 100  # You can adjust the number of epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model on the test dataset and save the best model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = model.state_dict()

# Save the final model and the best model
torch.save(model.state_dict(), 'souce_final_model.pth')
if best_model_state_dict:
    torch.save(best_model_state_dict, 'source_best_model.pth')

print('Training complete. Models saved.')
