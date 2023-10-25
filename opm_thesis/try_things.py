"""This script is used to classify the data using CSP and Logistic Regression.

This classifier takes the epochs from the different frequency bands and
acquisition times and trains a binary CSP classifier for each possible pair of
ids. Then, it predicts the label of each epoch using each classifier and gives the
results of the pairs.
"""
# import pickle
# import mne
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from mne.decoding import CSP
# from matplotlib import pyplot as plt

# from opm_thesis.preprocessing.utils import (
#     get_closest_sensors,
# )

# DATA_DIR = (
#     r"/Users/martin.iniguez/Desktop/master_thesis/"
#     r"opm-thesis/data/data_nottingham_preprocessed"
# )

# RESTING_PATH = DATA_DIR + "/resting_epochs.pkl"
# with open(RESTING_PATH, "rb") as f:
#     resting_epochs = pickle.load(f)
# info_resting = resting_epochs["alpha"].info

# ref_channels = []
# for ch in info_resting["chs"]:
#     if ch["kind"] == 301:
#         ref_channels.append(ch["ch_name"])


# alpha = dict({"l_freq": 8, "h_freq": 12})
# beta = dict({"l_freq": 12, "h_freq": 30})
# low_gamma = dict({"l_freq": 30, "h_freq": 60})
# high_gamma = dict({"l_freq": 60, "h_freq": 120})

# frequencies = {
#     "alpha": alpha,
#     "beta": beta,
#     "low_gamma": low_gamma,
#     "high_gamma": high_gamma,
# }


# # Define possible pairs for classification
# ids = [2**i for i in range(3, 8)]

# EPOCHS_PATH = DATA_DIR + "/all_epochs_filtered.pkl"
# with open(EPOCHS_PATH, "rb") as f:
#     epochs = pickle.load(f)

# total_scores = {}
# USE_HALF = True
# USE_X = False
# USE_REF = True

# KEY_FREQ = "alpha"
# epochs_freq = epochs[KEY_FREQ]
# resting_epochs_freq = resting_epochs[KEY_FREQ]

# indices = np.where(
#     epochs_freq.events[:, -1] == 8,
# )[0]

# all_epochs = epochs_freq[indices]
# picks = mne.pick_types(all_epochs.info, meg=True, exclude="bads")
# id_epochs = all_epochs.copy().pick(picks)
# resting_epochs = resting_epochs_freq.copy().pick(picks)

# if USE_X:
#     # selected_chs = [ch for ch in id_epochs.ch_names if '[X]' in ch]
#     selected_chs = [
#         ch for ch in get_closest_sensors(id_epochs.info, "LQ[X]", 27) if "[X]" in ch
#     ]
#     id_epochs = id_epochs.copy().pick(selected_chs)
#     resting_epochs = resting_epochs.copy().pick(selected_chs)

# if USE_REF:
#     selected_chs = [ch for ch in ref_channels if "[Y]" in ch]

#     id_epochs = all_epochs.copy().pick(ref_channels)
#     resting_epochs_freq = resting_epochs_freq.copy().pick(ref_channels)

# data_id = id_epochs.get_data()
# data_resting = resting_epochs_freq.get_data()

# # Using the inner half of the epochs (the second and third quarter) (1s)
# if USE_HALF:
#     quarter = data_id.shape[-1] // 4
#     data_id = data_id[:, :, quarter:]
#     data_id = data_id[:, :, :-quarter]

# data_epochs = np.concatenate((data_resting, data_id), axis=0)
# labels = np.concatenate(
#     (
#         np.ones(data_resting.shape[0]).astype(int),
#         np.zeros(data_id.shape[0]).astype(int),
#     )
# )

# # Building classifier
# clf = LinearDiscriminantAnalysis()

# # Split data into training and test sets once
# train_idx, test_idx = train_test_split(
#     np.arange(data_epochs.shape[0]),  # indices to split
#     test_size=0.2,  # 20% test size
#     random_state=50,  # seed for reproducibility
#     stratify=labels,  # preserve label balance
# )

# # Use the indices to select the train/test data
# X_train, y_train = data_epochs[train_idx], labels[train_idx]
# X_test, y_test = data_epochs[test_idx], labels[test_idx]

# # Fit CSP on training data
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False).fit(X_train, y_train)

# # Transform training and test data
# X_train_csp = csp.transform(X_train)
# X_test_csp = csp.transform(X_test)

# # Train and evaluate classifier
# clf.fit(X_train_csp, y_train)
# score = clf.score(X_test_csp, y_test)

# print(f"Score: {score:.2f}")

# if not USE_REF:
#     csp.plot_patterns(id_epochs.info, ch_type="mag", units="Patterns (AU)", size=1.5)
#     plt.show()


import pickle
import numpy as np
import mne

data_dir = "/Users/martin.iniguez/Desktop/master_thesis/opm-thesis/data/data_nottingham_preprocessed/preprocessing_163001.pkl"

with open(data_dir, "rb") as f:
    data = pickle.load(f)

raw = data.raw
raw.plot(block=True, events=data.events, event_id=data.event_id)
raw.save("163001_prepprocecssed_raw.fif")


# freqs = np.arange(5, 30)
# cycles = 1 / freqs
# tfr = mne.time_frequency.tfr_multitaper(
#     inst=epochs, return_itc=False, n_jobs=-1, freqs=freqs, n_cycles=freqs / 2, decim=2
# )

# tfr.plot_topo(baseline=(-1.5, -0.7), tmin=-1.5, tmax=1.5, dB=True)
