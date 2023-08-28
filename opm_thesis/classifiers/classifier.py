import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# Define a simple feedforward neural network
class Classifier(nn.Module):
    def __init__(self, num_channels, num_time_steps, batch_size, num_classes=5):
        super(Classifier, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps

        # 2D convolutional layers for spatial features
        self.conv1 = nn.Conv2d(batch_size, 32, kernel_size=(num_channels, 3))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2))

        # Fully connected layers for classification
        self.fc1 = nn.Linear(32 * (num_time_steps // 2), 128)  # Adjust as needed
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def train(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.to(dtype=torch.float32)  # Assuming you're using a GPU
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = self(batch_x.view(-1, batch_x.shape[1] * batch_x.shape[2]))
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")


# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
