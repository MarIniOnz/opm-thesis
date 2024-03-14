"""Self-written multi-class CSP classifier.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and
aggregates the predictions to get the final label."""

import pickle
from collections import Counter
import numpy as np
import mne

from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
ALL_EPOCHS_PATH = DATA_DIR + "/all_epochs_filtered.pkl"

with open(ALL_EPOCHS_PATH, "rb") as f:
    all_epochs = pickle.load(f)

# All possible pairs of ids
id_pairs = []
for i in range(5):
    for j in range(i + 1, 5):
        id_pairs.append([2 ** (i + 3), 2 ** (j + 3)])

# 1. Split the data into training and testing sets
labels = all_epochs[0].events[:, -1]  # Extracting labels from the events
picks = mne.pick_types(all_epochs[0].info, meg=True, exclude="bads")
data_epochs = [epochs.get_data()[:, picks, :] for epochs in all_epochs]

# Splitting the data indices into training and testing sets
train_indices, test_indices = train_test_split(
    np.arange(len(all_epochs[0])),
    test_size=0.2,  # Or whatever test size you want
    random_state=42,  # Use any seed for reproducibility
    stratify=labels,  # This ensures a balanced split
)

X_train = [epochs[train_indices] for epochs in data_epochs]
X_test = [epochs[test_indices] for epochs in data_epochs]

Y_train = labels[train_indices]
Y_test = labels[test_indices]

all_predictions = {}

for k, id_pair in enumerate(id_pairs):

    all_features_for_train = []
    all_features_for_test = []

    for freq_idx, (train_epochs, test_epochs) in enumerate(zip(X_train, X_test)):

        train_indices = np.where(
            np.logical_or(Y_train == id_pair[0], Y_train == id_pair[1])
        )[0]
        X_train_pair = train_epochs[train_indices]
        y_train_pair = Y_train[train_indices]

        test_indices = np.where(
            np.logical_or(Y_test == id_pair[0], Y_test == id_pair[1])
        )[0]
        X_test_pair = test_epochs[test_indices]
        y_test_pair = Y_test[test_indices]

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train_pair, y_train_pair)
        X_test_csp = csp.transform(X_test_pair)

        # Append the CSP-transformed data for the current frequency band to the list
        all_features_for_train.append(X_train_csp)
        all_features_for_test.append(X_test_csp)

    # Concatenate the CSP-transformed data across all frequency bands
    X_train_combined = np.concatenate(all_features_for_train, axis=1)
    X_test_combined = np.concatenate(all_features_for_test, axis=1)

    # Train the classifier using the combined features
    clf = LogisticRegression().fit(X_train_combined, y_train_pair)
    predictions = clf.predict(X_test_combined)

    for idx, pred in zip(test_indices, predictions):
        if idx not in all_predictions:
            all_predictions[idx] = []
        all_predictions[idx].append(pred)
    # all_predictions.append(predictions)

# 4. Aggregate Predictions
final_predictions = [-1] * len(Y_test)  # Initialize with -1 or some invalid label

for idx, predictions_for_idx in all_predictions.items():
    if Counter(predictions_for_idx).most_common(1)[0][1] > 2:
        most_common_label = Counter(predictions_for_idx).most_common(1)[0][0]
        final_predictions[idx] = most_common_label

# 5. Evaluate the final multi-class classifier
accuracy = accuracy_score(Y_test, final_predictions)

# Shuffle the final predictions to get the chance level
np.random.shuffle(final_predictions)
chance_level = accuracy_score(Y_test, final_predictions)
print(
    f"Final multi-class classifier accuracy: {accuracy:.3f} / Chance level: {chance_level:.3f}"
)
