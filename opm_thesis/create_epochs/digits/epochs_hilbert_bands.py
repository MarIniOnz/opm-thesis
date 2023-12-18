"""Hilbert transform of epochs in different frequency bands.

This script is used to create epochs in different frequency bands using the
Hilbert transform. The epochs are saved in different files for each frequency."""
import pickle
import os
import sys

import mne
import numpy as np

# Get triple parent directory of this file
path = os.path.abspath(__file__)
for _ in range(3):
    path = os.path.dirname(path)

sys.path.append(path)

DATA_SAVE = path + "/data/"
all_bads = []
acq_times = ["155445", "160513", "161344", "163001"]

all_data = dict({"l_freq": 0.01, "h_freq": 120})
alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12.5, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
mid_gamma = dict({"l_freq": 45, "h_freq": 75})
mid_high_gamma = dict({"l_freq": 60, "h_freq": 90})
high_gamma = dict({"l_freq": 60, "h_freq": 120})
all_gamma = dict({"l_freq": 30, "h_freq": 120})
mid_beta = dict({"l_freq": 15, "h_freq": 22})

frequencies = {
    "all_data": all_data,
    # "alpha": alpha,
    # "beta": beta,
    # "low_gamma": low_gamma,
    # "low_mid_gamma": low_mid_gamma,
    # "mid_gamma": mid_gamma,
    # "high_gamma": high_gamma,
    # "all_gamma": all_gamma,
    # "mid_high_gamma": mid_high_gamma,
    # "mid_beta": mid_beta,
}

for key, frequency_params in frequencies.items():
    preprocessed_epochs = []
    baseline = []

    for acq_idx, acq_time in enumerate(acq_times):
        with open(
            DATA_SAVE
            + "data_nottingham_preprocessed/analyzed/preprocessing_"
            + acq_time
            + ".pkl",
            "rb",
        ) as f:
            preprocessing = pickle.load(f)

        raw_filtered = preprocessing.apply_filters(
            preprocessing.raw_corrected,
            frequency_params,
            notch_filter=False,
        )
        picks = mne.pick_types(raw_filtered.info, meg="mag", exclude="bads")
        hilbert_transformed = raw_filtered.copy().apply_hilbert(picks=picks)
        hilbert_epochs = preprocessing.create_epochs(hilbert_transformed)
        baseline.append(hilbert_epochs.baseline[-1])

        preprocessed_epochs.append(hilbert_epochs)
        all_bads.extend(hilbert_epochs.info["bads"])

    baseline = np.mean(baseline)
    for acq_idx, epoch in enumerate(preprocessed_epochs):
        epoch.apply_baseline(baseline=(-2, baseline))
        epoch.info["bads"] = all_bads
        epoch.drop_bad()
        epoch.pick_types(meg="mag", exclude="bads")

    all_epochs = mne.concatenate_epochs(preprocessed_epochs)
    file_name = DATA_SAVE + "epochs/hilbert_" + key + "_all_epochs.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)

    all_epochs.decimate(4)
    file_name = DATA_SAVE + "epochs/hilbert_" + key + "_all_epochs_decimated.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)
