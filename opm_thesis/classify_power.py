"""Script to classify epochs using DeepConvNet."""

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from opm_thesis.classifiers.classifier import (
    DeepConvNet,
    MyDataset,
    TimeFreqCNN,
    VAR_CNN,
)


def predict(model, data_loader, device_to_use):
    """Predict the classes of the data in the data loader."""
    all_predictions = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device_to_use)
            outputs = model(batch_x)
            predicted = outputs.data.argmax(dim=1)
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_predictions)


frequencies = [
    "beta",
    # "all_data"
    # "low_freq"
    # "mid_freq",
    # "alpha",
    # "mid_beta",
    # "low_gamma",
    # "high_gamma",
    # "low_mid_gamma",
    # "mid_gamma",
    # "all_gamma",
]
# Define the path to the file
DATA_DIR = "./data/digits_epochs/freq_bands/"
FILENAME = DATA_DIR + frequencies[0] + "_all_epochs_decimated.pkl"
# FILENAME = DATA_DIR + "all_epochs.pkl"

# Generate all unique pairs of labels
labels_to_use = [32, 128]
# label_pairs = list(combinations(labels, 2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

NUM_SPLITS = 2
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=0)

# Load data
with open(FILENAME, "rb") as file:
    epochs = pickle.load(file)

# Using only one axis
USE_X = False
if USE_X:
    selected_chs = [ch for ch in epochs.ch_names if "[X]" in ch]
    epochs = epochs.pick("mag", selected_chs, type="mag")

epochs = epochs.pick("mag", exclude="bads")

valid_epochs_mask = np.isin(epochs.events[:, -1], labels_to_use)
data = np.real(epochs.get_data())[valid_epochs_mask]
data -= np.mean(data, axis=(0, 2), keepdims=True)
data /= np.std(data, axis=(0, 2), keepdims=True)

label_mapping = {label: idx for idx, label in enumerate(labels_to_use)}
labels = np.array(
    [label_mapping[label] for label in epochs.events[:, -1] if label in label_mapping]
)

# Store accuracies for each fold
accuracies = []

for train_index, test_index in kf.split(data):
    # Split data into training and test sets for this fold
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    # Create PyTorch datasets and loaders
    train_dataset = MyDataset(train_data, train_labels)
    test_dataset = MyDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize and train the model
    num_channels = train_data.shape[1]
    num_samples = train_data.shape[2]
    CLASSIFIER = DeepConvNet(num_channels, num_samples, len(labels_to_use), device).to(
        device
    )
    CLASSIFIER.train_model(
        train_loader,
        test_loader,
        num_epochs=100,
        learning_rate=1e-3,
    )

    # Evaluate the model
    accuracy = CLASSIFIER.evaluate(test_loader) * 100
    accuracies.append(accuracy)
    print(f"Accuracy for fold: {accuracy:.2f}")

# Calculate and print the mean accuracy
mean_accuracy = np.mean(accuracies)
# Print the mean accuracy rounded to 2 decimal places
print(f"Mean accuracy: {mean_accuracy:.2f}")
