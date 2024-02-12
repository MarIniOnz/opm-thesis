"""Deep Convolutional Network for OPM Classification, based on the paper:

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter,
M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning
with convolutional neural networks for EEG decoding and visualization.
Human brain mapping, 38(11), 5391-5420.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader


class DeepConvNet(nn.Module):
    """Deep Convolutional Network for OPM Classification."""

    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        num_classes: int = 5,
        device: str = "cpu",
    ):
        super(DeepConvNet, self).__init__()

        self.device = device
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input data of shape (n_samples, 1, n_channels, n_timesteps)
        :type x: torch.Tensor
        :return: Output of the model
        :rtype: torch.Tensor
        """
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
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """Train the model.

        :param train_loader: Training data loader
        :type train_loader: DataLoader
        :param test_loader: Test data loader
        :type test_loader: DataLoader
        :param num_epochs: Number of epochs to train the model, defaults to 10
        :type num_epochs: int, optional
        :param learning_rate: Learning rate, defaults to 0.001
        :type learning_rate: float, optional
        """
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

            # Evaluate and print test error every second epoch
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1:
                self.eval()  # Set the model to evaluation mode
                accuracy = self.evaluate(test_loader) * 100
                print(
                    f"Epoch [{epoch+1}/{num_epochs}],  Loss: {loss.item():.4f},  Test Accuracy: {accuracy:.2f}"
                )

    def evaluate(self, test_loader: DataLoader):
        """Evaluate the model.

        :param test_loader: Test data loader
        :type test_loader: DataLoader
        :return: Accuracy of the model
        :rtype: float
        """
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

    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        num_classes: int = 5,
        device: str = "cpu",
    ):
        super(SimplifiedDeepConvNet, self).__init__(
            num_channels, num_samples, num_classes
        )
        self.device = device

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input data of shape (n_samples, 1, n_channels, n_timesteps)
        :type x: torch.Tensor
        :return: Output of the model
        :rtype: torch.Tensor
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class TimeFreqCNN(nn.Module):
    """Time-Frequency Convolutional Neural Network for OPM Classification."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_freqs: int,
        num_timepoints: int,
        device: str = "cpu",
    ):
        """Initialize the model.

        :param num_classes: Number of classes
        :type num_classes: int
        :param input_size: Number of time points in the input data
        :type input_size: int
        """
        super(TimeFreqCNN, self).__init__()
        self.device = device

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
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

        # Calculate the size of the features after convolutional and pooling layers
        reduced_freqs = num_freqs // (2**3)  # Assuming 3 pooling layers
        reduced_timepoints = num_timepoints // (2**3)  # Assuming 3 pooling layers
        self.flattened_size = 64 * reduced_freqs * reduced_timepoints

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input data of shape (n_samples, 1, n_channels, n_timesteps)
        :type x: torch.Tensor
        :return: Output of the model
        :rtype: torch.Tensor
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x.squeeze(1)))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

    def train_model(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """Train the model.

        :param train_loader: Training data loader
        :type train_loader: DataLoader
        :param num_epochs: Number of epochs to train the model, defaults to 10
        :type num_epochs: int, optional
        :param learning_rate: Learning rate, defaults to 0.001
        :type learning_rate: float, optional
        """
        self.train()  # Set the model to training mode
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                batch_x = inputs.to(self.device)
                batch_y = labels.to(self.device)

                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {running_loss / i:.4f}")

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate the model.

        :param test_loader: Test data loader
        :type test_loader: DataLoader
        :return: Accuracy of the model (between 0 and 1)
        :rtype: float
        """
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                batch_x = inputs.to(self.device)
                batch_y = labels.to(self.device)
                outputs = self(batch_x)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        return accuracy


class SpatialDeMixing(nn.Module):
    """Spatial De-Mixing Layer for OPM Classification."""

    def __init__(self, input_channels: int, output_channels: int):
        super(SpatialDeMixing, self).__init__()
        self.output_channels = output_channels
        self.spatial_filters = nn.Linear(
            input_channels, self.output_channels, bias=False
        )

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        :param x: Input data of shape (batch_size, 1, n_channels, input_size)
        :type x: torch.Tensor
        :return: Output of the model
        :rtype: torch.Tensor
        """

        # x should be of shape (batch_size, 1, channels, time_points)
        # We will apply the spatial filters across the channels for each time point
        batch_size, _, time_points = x.shape
        x = x.view(-1, x.size(1))
        x = self.spatial_filters(x)  # Apply the spatial filters
        x = x.view(
            batch_size, self.output_channels, time_points
        )  # Reshape to the original dimensions with new channels
        return x


class LF_CNN(nn.Module):
    """VAR-CNN for OPM Classification.

    Based on the paper:
    Adaptive neural network classifier for decoding MEG signals
    """

    def __init__(
        self,
        input_channels: int,
        input_size: int,
        num_classes: int,
        device,
    ):
        super(LF_CNN, self).__init__()
        self.device = device
        self.num_spatial_filters = 32
        length_temporal_filter = 7

        # Define the spatial de-mixing layer
        self.spatial_demixing = SpatialDeMixing(
            input_channels=input_channels, output_channels=self.num_spatial_filters
        )  # Number of latent sources 'k'

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Define the 2D convolutional layert
        self.temporal_convolution = nn.Conv1d(
            in_channels=self.num_spatial_filters,
            out_channels=self.num_spatial_filters,
            kernel_size=length_temporal_filter,
            stride=1,
            padding=(length_temporal_filter - 1) // 2,
        )  # Adjust padding to maintain size

        # Define the max pooling layer
        # If you want to apply max pooling only to the time dimension
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Define the fully connected layer
        num_inputs = self.calculate_feature_map_size(input_size, input_channels)
        self.fc = nn.Linear(
            in_features=num_inputs,
            out_features=num_classes,
        )

    def calculate_feature_map_size(self, input_size: int, input_channels: int) -> int:
        """Calculate the size of the feature map after the convolutional layers.

        :param input_size: Number of time points in the input data
        :type input_size: int
        :param input_channels: Number of channels in the input data
        :type input_channels: int
        :return: Number of elements in the feature map
        :rtype: int
        """
        # Simulate the forward pass up to the point before the fully connected layer
        # to determine the correct number of input features
        with torch.no_grad():
            # Create a dummy tensor with the expected input shape
            x = torch.zeros((1, input_channels, input_size))

            # Apply the spatial demixing
            x = self.spatial_demixing(x)

            # Apply the 2D convolution
            x = self.temporal_convolution(x)
            # Apply the max pooling layer
            x = self.maxpool(x)
            # Flatten the output
            return x.numel()  # Number of elements in the tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        :param x: Input data of shape (batch_size, 1, n_channels, input_size)
        :type x: torch.Tensor
        :return: Output of the model
        :rtype: torch.Tensor
        """
        # Apply the spatial de-mixing layer
        x = self.spatial_demixing(x)
        # Apply the 2D convolution
        x = F.relu(self.temporal_convolution(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply the max pooling layer
        x = self.maxpool(x)
        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer
        x = self.fc(x)

        return x

    def train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 3e-4,
    ) -> None:
        """Train the model.

        :param train_loader: Training data loader
        :type train_loader: DataLoader
        :param test_loader: Test data loader
        :type test_loader: DataLoader
        :param num_epochs: Number of epochs to train the model, defaults to 10
        :type num_epochs: int, optional
        :param learning_rate: Learning rate, defaults to 0.001
        :type learning_rate: float, optional
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_x.squeeze())
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluate and print test error every 100 epoch
            if (epoch + 1) % 100 == 0:
                self.eval()  # Set the model to evaluation mode
                accuracy = self.evaluate(test_loader) * 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}")

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate the model.

        :param test_loader: Test data loader
        :type test_loader: DataLoader
        :return: Accuracy of the model
        :rtype: float
        """
        correct = 0
        total = 0
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self(batch_x.squeeze())
                probabilities = F.softmax(outputs, dim=1)  # Apply softmax
                predicted = probabilities.argmax(dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy


# Define a custom dataset
class MyDataset(Dataset):
    """Custom Dataset for loading OPM data."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
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
