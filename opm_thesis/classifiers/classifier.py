import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class Classifier(nn.Module):
    def __init__(self, output_size=5):
        super(Classifier, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network. Returns the output of the network.
        Must be defined in subclasses.

        :param x: Input data
        :type x: torch.Tensor
        :return: Output of the network
        :rtype: torch.Tensor
        """
        pass

    def train(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x.view(-1, batch_x.shape[1] * batch_x.shape[2]))
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
