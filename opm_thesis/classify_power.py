import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

from opm_thesis.classifiers.classifier import DeepConvNet, MyDataset

path = os.path.abspath(__file__)
for _ in range(3):
    path = os.path.dirname(path)

sys.path.append(path)


def predict(model, data_loader, device):
    """Predict the classes of the data in the data loader."""
    all_predictions = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predicted = outputs.data.argmax(dim=1)
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_predictions)


# Define the path to the file
DATA_DIR = "./data/epochs"
FILENAME = DATA_DIR + "/hilbert_alpha_all_epochs_decimated.pkl"

# Generate all unique pairs of labels
labels_to_use = [8, 16, 32, 64, 128]
# label_pairs = list(combinations(labels, 2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary to store accuracies
accuracies = {}

# for labels_to_use in label_pairs:
with open(FILENAME, "rb") as file:
    epochs = pickle.load(file)

valid_epochs_mask = np.isin(epochs.events[:, -1], labels_to_use)
data = epochs.get_data()[valid_epochs_mask]

# Normalize the data per channel and take only real part
data = np.real(data)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

# Map labels to 0 and 1
label_mapping = {label: idx for idx, label in enumerate(labels_to_use)}
labels = np.array(
    [label_mapping[label] for label in epochs.events[:, -1] if label in label_mapping]
)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# Create PyTorch datasets and loaders
train_dataset = MyDataset(train_data, train_labels)
test_dataset = MyDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize the model
num_channels = train_data.shape[1]
num_samples = train_data.shape[2]
classifier = DeepConvNet(num_channels, num_samples, len(labels_to_use)).to(
    device
)  # Two classes

classifier.train(train_loader, num_epochs=150, learning_rate=1e-3)
accuracy = classifier.evaluate(test_loader)
print(f"Accuracy for labels {labels_to_use}: {accuracy}")
