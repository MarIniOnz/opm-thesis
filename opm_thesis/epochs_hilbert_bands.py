"""Hilbert transform of epochs in different frequency bands.

This script is used to create epochs in different frequency bands using the
Hilbert transform. The epochs are saved in different files for each frequency."""
import pickle
import mne


DATA_SAVE = (
    r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
)

all_bads = []
acq_times = ["155445", "160513", "161344", "163001"]

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12.5, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
mid_gamma = dict({"l_freq": 45, "h_freq": 75})
high_gamma = dict({"l_freq": 60, "h_freq": 120})
all_gamma = dict({"l_freq": 30, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "low_mid_gamma": low_mid_gamma,
    "mid_gamma": mid_gamma,
    "high_gamma": high_gamma,
    "all_gamma": all_gamma,
}

for key, frequency_params in frequencies.items():
    preprocessed_epochs = []

    for acq_idx, acq_time in enumerate(acq_times):
        with open(DATA_SAVE + "\\preprocessing_" + acq_time + ".pkl", "rb") as f:
            preprocessing = pickle.load(f)

        raw_filtered = preprocessing.apply_filters(
            preprocessing.raw,
            frequency_params,
            notch_filter=False,
        )
        picks = mne.pick_types(raw_filtered.info, meg="mag", exclude="bads")
        hilbert_transformed = raw_filtered.copy().apply_hilbert(picks=picks)
        hilbert_epochs = preprocessing.create_epochs(hilbert_transformed)

        preprocessed_epochs.append(hilbert_epochs)
        all_bads.extend(hilbert_epochs.info["bads"])

    for acq_idx, epoch in enumerate(preprocessed_epochs):
        epoch.info["bads"] = all_bads

    all_epochs = mne.concatenate_epochs(preprocessed_epochs)
    file_name = DATA_SAVE + "\\epochs\\hilbert_" + key + "_all_epochs.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)