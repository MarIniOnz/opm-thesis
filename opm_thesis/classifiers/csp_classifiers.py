"""This script is used to classify the data using CSP and Logistic Regression.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and gives the
results of the pairs.
"""
import pickle
import mne
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne.decoding import CSP

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
high_gamma = dict({"l_freq": 60, "h_freq": 120})

# low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
# mid_gamma = dict({"l_freq": 45, "h_freq": 75})
# all_gamma = dict({"l_freq": 30, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "high_gamma": high_gamma,
    # "low_mid_gamma": low_mid_gamma,
    # "mid_gamma": mid_gamma,
    # "all_gamma": all_gamma,
}

acq_times = ["155445", "160513", "161344", "163001"]
DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
# Write all possible pair combinations of 8, 16, 32, 64, 128
id_pairs = []
for i in range(5):
    for j in range(i + 1, 5):
        id_pairs.append([2 ** (i + 3), 2 ** (j + 3)])

EPOCHS_PATH = DATA_DIR + "/all_epochs_filtered.pkl"
with open(EPOCHS_PATH, "rb") as f:
    epochs = pickle.load(f)

total_scores = {}
use_half = True

for pair_idx, id_pair in enumerate(id_pairs):
    labels = []
    csp_list = []
    data_list = []

    for key_idx, (key, frequency_params) in enumerate(frequencies.items()):

        epochs_freq = epochs[key_idx]
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
        if use_half:
            quarter = data_epochs.shape[-1] // 4
            data_epochs = data_epochs[:, :, quarter:]
            data_epochs = data_epochs[:, :, :-quarter]

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        data_csp = csp.fit_transform(data_epochs, pair_epochs.events[:, -1])

        labels.append(pair_epochs.events[:, -1])
        data_list.append(data_csp)
        csp_list.append(csp)

    data_concat = np.concatenate(data_list)
    labels_concat = np.concatenate(labels)

    # Building classifier
    clf = LinearDiscriminantAnalysis()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    scores = cross_val_score(clf, data_concat, labels_concat, cv=cv, n_jobs=1)
    total_scores[f"{id_pair[0]} and {id_pair[1]}"] = scores.mean()
    print(
        f"Classification accuracy {id_pair[0]} and {id_pair[1]}: "
        f"{scores.mean():.3f} / Chance level: 0.5"
    )

print(total_scores)
