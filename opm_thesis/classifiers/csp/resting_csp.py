"""This script is used to classify the data using CSP and Logistic Regression.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and gives the
results of the pairs.
"""
import pickle
import mne
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
high_gamma = dict({"l_freq": 60, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "high_gamma": high_gamma,
}

DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)

# Define possible pairs for classification
ids = [2**i for i in range(3, 8)]

EPOCHS_PATH = DATA_DIR + "/all_epochs_filtered.pkl"
with open(EPOCHS_PATH, "rb") as f:
    epochs = pickle.load(f)

RESTING_PATH = DATA_DIR + "/resting_epochs.pkl"
with open(RESTING_PATH, "rb") as f:
    resting_epochs = pickle.load(f)

total_scores = {}
USE_HALF = True

for pair_idx, id_compared in enumerate(ids):
    labels = []
    csp_list = []
    data_list = []

    for key, frequency_params in frequencies.items():

        epochs_freq = epochs[key]
        resting_epochs_freq = resting_epochs[key]

        indices = np.where(
            epochs_freq.events[:, -1] == id_compared,
        )[0]

        id_epochs = epochs_freq[indices]
        picks = mne.pick_types(id_epochs.info, meg=True, exclude="bads")

        data_resting = resting_epochs_freq.get_data()[:, picks, :]
        data_id = id_epochs.get_data()[:, picks, :]

        # Using the inner half of the epochs (the second and third quarter) (1s)
        if USE_HALF:
            quarter = data_id.shape[-1] // 4
            data_id = data_id[:, :, quarter:]
            data_id = data_id[:, :, :-quarter]

        data_epochs = np.concatenate((data_resting, data_id), axis=0)
        labels_concat = np.concatenate(
            (
                np.ones(data_resting.shape[0]).astype(int),
                np.zeros(data_id.shape[0]).astype(int),
            )
        )

        labels.append(labels_concat)
        data_list.append(data_epochs)

    # Building classifier and cross-validator
    clf = LinearDiscriminantAnalysis()
    cv = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)

    # Initialize a list to store scores for each fold
    scores = []

    # Custom cross-validation loop
    for train_idx, test_idx in cv.split(data_list[0], labels_concat):
        train_data_csp = []
        test_data_csp = []

        # Iterate through each frequency band
        for data_epochs in data_list:
            # Split data into training and test sets
            X_train, y_train = data_epochs[train_idx], labels_concat[train_idx]
            X_test, y_test = data_epochs[test_idx], labels_concat[test_idx]

            # Fit CSP on training data
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False).fit(
                X_train, y_train
            )

            # Transform training and test data and append to lists
            train_data_csp.append(csp.transform(X_train))
            test_data_csp.append(csp.transform(X_test))

        # Concatenate transformed data from all frequency bands
        X_train_csp = np.hstack(train_data_csp)
        X_test_csp = np.hstack(test_data_csp)

        # Train and evaluate classifier
        clf.fit(X_train_csp, y_train)
        score = clf.score(X_test_csp, y_test)
        scores.append(score)

    total_scores[f"{id_compared} and resting"] = np.mean(scores)

print(total_scores)
