"""Creates epochs for each frequency band and acquisition time.

This script creates epochs for each frequency band and acquisition time. The epochs
are saved in a list of lists of lists. The first index corresponds to the frequency
band, the second index corresponds to the acquisition time, and the third index
corresponds to the epoch.
"""
import pickle
import mne
import numpy as np

acq_times = ["155445", "160513", "161344", "163001"]
DATA_SAVE = "./data/"

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
high_gamma = dict({"l_freq": 60, "h_freq": 120})

low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
mid_gamma = dict({"l_freq": 45, "h_freq": 75})
all_gamma = dict({"l_freq": 30, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    # "low_gamma": low_gamma,
    # "high_gamma": high_gamma,
    # "low_mid_gamma": low_mid_gamma,
    # "mid_gamma": mid_gamma,
    # "all_gamma": all_gamma,
}


for key_idx, (key, frequency_params) in enumerate(frequencies.items()):
    freq_epochs = []
    baseline = []
    all_bads = []

    for acq_idx, acq_time in enumerate(acq_times):
        with open(
            DATA_SAVE
            + "/data_nottingham_preprocessed/analyzed/preprocessing_"
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
        epochs = preprocessing.create_epochs(raw_filtered)
        freq_epochs.append(epochs)
        baseline.append(epochs.baseline[-1])

        all_bads.extend(epochs.info["bads"])

    baseline = np.mean(baseline)
    for acq_idx, epoch in enumerate(freq_epochs):
        epoch.apply_baseline(baseline=(-2, baseline))
        epoch.info["bads"] = all_bads
        epoch.drop_bad()
        epoch.pick_types(meg="mag", exclude="bads")

    all_epochs = mne.concatenate_epochs(freq_epochs)
    file_name = DATA_SAVE + "epochs/freq_bands/" + key + "_all_epochs.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)

    all_epochs.decimate(4)
    file_name = DATA_SAVE + "epochs/freq_bands/" + key + "_all_epochs_decimated.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)
