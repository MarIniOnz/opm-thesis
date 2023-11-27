"""Deep Convolutional Network for OPM Classification, based on the paper:

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter,
M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning
with convolutional neural networks for EEG decoding and visualization.
Human brain mapping, 38(11), 5391-5420.
"""
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset


class DeepConvNet(nn.Module):
    """Deep Convolutional Network for OPM Classification."""

    def __init__(self, num_channels: int, num_samples: int, num_classes: int = 5):
        super(DeepConvNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, 25, (1, 10), padding=(0, 5))

        # Spatial Convolution
        self.spatial_conv = nn.Conv2d(25, 25, (num_channels, 1))

        # Batch Normalization, ReLU, MaxPooling and Dropout
        self.batch_norm1 = nn.BatchNorm2d(25)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((1, 3))
        self.dropout1 = nn.Dropout(0.5)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(25, 50, (1, 10), padding=(0, 5))
        self.batch_norm2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d((1, 3))
        self.dropout2 = nn.Dropout(0.5)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(50, 100, (1, 10), padding=(0, 5))
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.maxpool3 = nn.MaxPool2d((1, 3))
        self.dropout3 = nn.Dropout(0.5)

        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(100, 200, (1, 10), padding=(0, 5))
        self.batch_norm4 = nn.BatchNorm2d(200)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d((1, 3))
        self.dropout4 = nn.Dropout(0.5)

        # Classification Layer
        self.fc1 = nn.Linear(
            200 * ((num_samples // (3**4)) + 1), num_classes
        )  # Size of the data based on the pooling and convolutional layers
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

    def train_model(
        self, train_loader, test_loader, num_epochs=10, learning_rate=0.001
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Evaluate and print test error every second epoch
            if (epoch + 1) % 5 == 0:
                self.eval()  # Set the model to evaluation mode
                test_accuracy = self.evaluate(test_loader)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}"
                )

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self(batch_x)
                predicted = outputs.data.argmax(dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy


class SimplifiedDeepConvNet(DeepConvNet):
    """Simplified Deep Convolutional Network for OPM Classification."""

    def __init__(self, num_channels: int, num_samples: int, num_classes: int = 5):
        super(SimplifiedDeepConvNet, self).__init__(
            num_channels, num_samples, num_classes
        )

        # First Convolutional Block (Temporal then Spatial Convolution)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), padding=(0, 2)),  # Temporal Convolution
            nn.Conv2d(25, 25, (num_channels, 1)),  # Spatial Convolution
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # MaxPooling
            nn.Dropout(0.5),
        )

        # Second Convolutional Block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

        # Classification Layer
        self.fc1 = nn.Linear(
            50 * ((num_samples // 4)), num_classes  # Adjust the size accordingly
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class TimeFreqCNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super(TimeFreqCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Calculate the size of the flattened features after convolutional and pooling layers
        self.flattened_size = 64 * (input_size // 8) * (input_size // 8)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        self.train()  # Set the model to training mode
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0

            print(f"Epoch {epoch + 1} completed")

    def evaluate(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy


# Define a custom dataset
class MyDataset(Dataset):
    """Custom Dataset for loading OPM data."""

    def __init__(self, data, labels):
        data = torch.from_numpy(data).type(torch.float32)
        data = data.unsqueeze(
            1
        )  # Data shape should be (n_samples, 1, n_channels, n_timesteps)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
