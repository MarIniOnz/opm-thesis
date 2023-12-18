"""Hilbert transform of epochs in different frequency bands.

This script is used to create epochs in different frequency bands using the
Hilbert transform. The epochs are saved in different files for each frequency."""
import pickle

import mne

DATA_SAVE = "./data/"
all_bads = []

# Frequencies in Hz
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
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "low_mid_gamma": low_mid_gamma,
    "mid_gamma": mid_gamma,
    "high_gamma": high_gamma,
    "all_gamma": all_gamma,
    "mid_high_gamma": mid_high_gamma,
    "mid_beta": mid_beta,
}

for key, frequency_params in frequencies.items():
    preprocessed_epochs = []

    for num in range(4):
        with open(
            DATA_SAVE
            + "gestures/preprocessed/preprocessing_gestures_"
            + str(num + 1)
            + ".pkl",
            "rb",
        ) as f:
            preprocessing = pickle.load(f)

        raw_filtered = preprocessing.apply_filters(
            preprocessing.raw,
            frequency_params,
            notch_filter=False,
        )
        picks = mne.pick_types(raw_filtered.info, meg="mag", exclude="bads")
        hilbert_transformed = raw_filtered.copy().apply_hilbert(picks=picks)
        hilbert_epochs = preprocessing.create_epochs(hilbert_transformed, gestures=True)

        preprocessed_epochs.append(hilbert_epochs)
        all_bads.extend(hilbert_epochs.info["bads"])

    for acq_idx, epoch in enumerate(preprocessed_epochs):
        epoch.info["bads"] = all_bads
        epoch.drop_bad()
        epoch.pick_types(meg="mag", exclude="bads")

    all_epochs = mne.concatenate_epochs(preprocessed_epochs)
    FILE_NAME = DATA_SAVE + "gestures_epochs/hilbert/" + key + "_all_epochs.pkl"

    with open(FILE_NAME, "wb") as f:
        pickle.dump(all_epochs, f)

    all_epochs.decimate(4)
    FILE_NAME = (
        DATA_SAVE + "gestures_epochs/hilbert/" + key + "_all_epochs_decimated.pkl"
    )

    with open(FILE_NAME, "wb") as f:
        pickle.dump(all_epochs, f)
