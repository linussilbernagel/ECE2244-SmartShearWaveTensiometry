import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import pandas as pd
from sklearn.model_selection import train_test_split

# Define your classification CNN model
class WaveSpeedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WaveSpeedClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # Adjust the spatial size for AlexNet's fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define your classification CNN model
class AlexNet1D(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(AlexNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6)  # Reshape for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load pre-trained AlexNet model
pretrained_alexnet = models.alexnet(pretrained=True)

# Freeze all layers except the final fully connected layers
for param in pretrained_alexnet.parameters():
    param.requires_grad = False

# Modify the final fully connected layers to match the number of classes in your classification task
num_classes = 4  # Adjust according to your specific task
pretrained_alexnet.classifier = nn.Sequential(
    nn.Linear(9216, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
)

# Load your data
dataset = pd.read_csv('./Code/data.csv', header=None, dtype=float)
label = pd.read_csv('./Code/labels.csv', header=None, dtype=int)

x_train, x_test, y_train, y_test = train_test_split(dataset.values, label.values.T - 1, test_size=0.2, random_state=42) 
# Assuming x_train, y_train, x_val, y_val are numpy arrays
# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Assuming labels are integers
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_test_tensor = torch.tensor(y_test, dtype=int)

# Create DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize your model
model = AlexNet1D(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
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
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Test the model (not shown in this example)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.models import alexnet

# # Load the pretrained AlexNet model
# pretrained_model = alexnet(pretrained=True)

# # Check the architecture of the pretrained model
# print(pretrained_model)

# # Modify the first convolutional layer for 1D input
# pretrained_model.features[0] = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2)

# # Freeze the weights of the modified layer
# for param in pretrained_model.features[0].parameters():
#     param.requires_grad = False

# # Modify the last fully connected layer for your specific task
# num_classes = 10  # Adjust according to your task
# pretrained_model.classifier[6] = nn.Linear(4096, num_classes)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

# # Training loop (not shown in this example)
