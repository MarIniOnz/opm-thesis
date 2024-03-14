"""This script is used to classify the data using CSP and LDA.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and gives the
results of the pairs.
"""

import pickle
import mne
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
low_mid_gamma = dict({"l_freq": 30, "h_freq": 90})
mid_gamma = dict({"l_freq": 60, "h_freq": 90})
high_gamma = dict({"l_freq": 60, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "low_mid_gamma": low_mid_gamma,
    "mid_gamma": mid_gamma,
    "high_gamma": high_gamma,
}

DATA_DIR = "./data/epochs/freq_bands/"

# Define possible pairs for classification
id_pairs = [[2**i, 2**j] for i in range(3, 8) for j in range(i + 1, 8)]
id_pairs = [[16, 128]]

total_scores = {}
USE_HALF = False

for pair_idx, id_pair in enumerate(id_pairs):
    labels = []
    csp_list = []
    data_list = []

    for key, frequency_params in frequencies.items():
        with open(DATA_DIR + f"{key}_all_epochs_decimated.pkl", "rb") as f:
            epochs_freq = pickle.load(f)
        indices = np.where(
            np.logical_or(
                epochs_freq.events[:, -1] == id_pair[0],
                epochs_freq.events[:, -1] == id_pair[1],
            )
        )[0]
        pair_epochs = epochs_freq[indices]

        picks = mne.pick_types(pair_epochs.info, meg=True, exclude="bads")
        data_epochs = pair_epochs.get_data()[:, picks, :]

        # Using the inner half of the epochs (the second and third quarter) (1s)
        if USE_HALF:
            quarter = data_epochs.shape[-1] // 4
            data_epochs = data_epochs[:, :, quarter:]
            data_epochs = data_epochs[:, :, :-quarter]

        labels.append(pair_epochs.events[:, -1])
        data_list.append(data_epochs)

    labels_concat = np.concatenate(labels)

    # Building classifier and cross-validator
    clf = LinearDiscriminantAnalysis()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # Initialize a list to store scores for each fold
    scores = []

    # Custom cross-validation loop
    for train_idx, test_idx in cv.split(data_list[0]):
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

    total_scores[f"{id_pair[0]} and {id_pair[1]}"] = np.mean(scores)

print(total_scores)
