import sys
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

path = os.path.abspath(__file__)
for _ in range(2):
    path = os.path.dirname(path)

sys.path.append(path)

from opm_thesis.classifiers.classifier import DeepConvNet, MyDataset


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
FILENAME = DATA_DIR + "/hilbert_low_gamma_all_epochs_decimated.pkl"

# Generate all unique pairs of labels
labels_to_use = [8, 128]
# label_pairs = list(combinations(labels, 2))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Load data
with open(FILENAME, "rb") as file:
    epochs = pickle.load(file)

valid_epochs_mask = np.isin(epochs.events[:, -1], labels_to_use)
data = np.real(epochs.get_data())[
    valid_epochs_mask
]  # Normalize and take only real part
# data -= np.mean(data, axis=2, keepdims=True)
# data /= np.std(data, axis=2, keepdims=True)

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
    classifier = DeepConvNet(num_channels, num_samples, len(labels_to_use)).to(device)
    # classifier = TimeFreqCNN(len(labels_to_use)).to(device)
    classifier.train_model(train_loader, test_loader, num_epochs=75, learning_rate=1e-3)

    # Evaluate the model
    accuracy = classifier.evaluate(test_loader)
    accuracies.append(accuracy)
    print(f"Accuracy for fold: {accuracy}")

# Calculate and print the mean accuracy
mean_accuracy = np.mean(accuracies)
print(f"Mean accuracy over {n_splits} folds: {mean_accuracy}")
