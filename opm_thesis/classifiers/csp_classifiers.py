"""This script is used to classify the data using CSP and Logistic Regression.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and gives the
results of the pairs.
"""
import pickle
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score

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

all_epochs = [[[] for _ in range(4)] for _ in range(len(id_pairs))]
data_list = [[] for _ in range(len(id_pairs))]
csp_list = [[] for _ in range(len(id_pairs))]
labels = [[] for _ in range(len(id_pairs))]

all_bads = []

for key_idx, (key, frequency_params) in enumerate(frequencies.items()):
    for acq_idx, acq_time in enumerate(acq_times):
        with open(DATA_DIR + "/preprocessing_" + acq_time + ".pkl", "rb") as f:
            preprocessing = pickle.load(f)
        raw_filtered = preprocessing.apply_filters(
            preprocessing.raw,
            frequency_params,
            notch_filter=False,
        )
        epochs = preprocessing.create_epochs(raw_filtered)

        for pair_idx, id_pair in enumerate(id_pairs):
            indices = np.where(
                np.logical_or(
                    epochs.events[:, -1] == id_pair[0],
                    epochs.events[:, -1] == id_pair[1],
                )
            )[0]
            all_epochs[pair_idx][acq_idx] = epochs[indices]

        all_bads.extend(epochs.info["bads"])

    for k in range(len(id_pairs)):
        for acq_idx in range(4):
            all_epochs[k][acq_idx].info["bads"] = all_bads

        concatenated = mne.concatenate_epochs(all_epochs[k][:])
        picks = mne.pick_types(concatenated.info, meg=True, exclude="bads")
        data_epochs = concatenated.get_data()[:, picks, :]

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        data_csp = csp.fit_transform(data_epochs, concatenated.events[:, -1])

        labels[k].append(concatenated.events[:, -1])
        data_list[k].append(data_csp)
        csp_list[k].append(csp)

total_scores = {}
for k, id_pair in enumerate(id_pairs):
    data_concat = np.concatenate(data_list[k])
    labels_concat = np.concatenate(labels[k])

    clf = LogisticRegression()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, data_concat, labels_concat, cv=cv, n_jobs=1)
    total_scores[f"{id_pair[0]} and {id_pair[1]}"] = scores.mean()
    print(
        f"Classification accuracy {id_pair[0]} and {id_pair[1]}: "
        f"{scores.mean():.3f} / Chance level: 0.5"
    )

print(total_scores)