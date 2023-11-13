"""Classify epochs using a DeepConvNet classifier."""
import pickle

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from opm_thesis.classifiers.classifier import DeepConvNet, MyDataset

# Define the path to the file
DATA_DIR = "./data/epochs"
FILENAME = DATA_DIR + "/hilbert_mid_beta_all_epochs_decimated.pkl"

# Concatenate all the data into a single array
labels_to_use = [16, 64]

with open(FILENAME, "rb") as FILENAME:
    epochs = pickle.load(FILENAME)
valid_epochs_mask = np.isin(epochs.events[:, -1], labels_to_use)
data = epochs.get_data()[valid_epochs_mask]

# Normalize the data per channel and take only real part
data = np.real(data)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

# For two classes
label_mapping = {label: idx for idx, label in enumerate(labels_to_use)}
labels = np.array(
    [label_mapping[label] for label in epochs.events[:, -1] if label in label_mapping]
)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Create PyTorch datasets and loaders
train_dataset = MyDataset(train_data, train_labels)
test_dataset = MyDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Create an instance of the classifier
num_channels = train_data.shape[1]
num_samples = train_data.shape[2]
num_classes = len(np.unique(labels))
classifier = DeepConvNet(num_channels, num_samples, num_classes)

# Train the classifier and evaluate it
classifier.train(train_loader, num_epochs=100, learning_rate=1e-4)
classifier.evaluate(test_loader)


# Predictions
def predict(model, data_loader):
    """Predict the classes of the data in the data loader."""
    all_predictions = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            outputs = model(batch_x)
            predicted = outputs.data.argmax(dim=1)
            all_predictions.extend(predicted.numpy())
    return np.array(all_predictions)


predictions = predict(classifier, test_loader)

# Print the predictions
print(predictions)
