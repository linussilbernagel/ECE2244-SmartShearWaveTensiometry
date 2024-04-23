import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define your CNN architecture
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(15488, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        batch_size = x.size(0)  # Get the batch size dynamically
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
dataset = pd.read_csv('./Code/data.csv', header=None, dtype=float)
label = pd.read_csv('./Code/labels.csv', header=None, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(dataset.values, label.values.T - 1, test_size=0.2, random_state=42) 
# y_train = y_train - 1
# y_test = y_test - 1

# # Example usage
num_classes = 4  # Define the number of classes in your classification task

# # Create an instance of the CNN model
model = CNN((500,1), num_classes=num_classes)
# model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert your data into PyTorch tensors and create DataLoader
# Assuming x_train, y_train, x_val, y_val, x_test, y_test are numpy arrays
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=int)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=int)

# Flatten the labels tensor
y_train_tensor = torch.flatten(y_train_tensor)
y_test_tensor = torch.flatten(y_test_tensor)


train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # PyTorch expects input in (batch_size, channels, sequence_length) format
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')
