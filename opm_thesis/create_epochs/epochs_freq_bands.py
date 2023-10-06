"""Creates epochs for each frequency band and acquisition time.

This script creates epochs for each frequency band and acquisition time. The epochs
are saved in a list of lists of lists. The first index corresponds to the frequency
band, the second index corresponds to the acquisition time, and the third index
corresponds to the epoch.
"""
import pickle
import mne

acq_times = ["155445", "160513", "161344", "163001"]
DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
ALL_EPOCHS_PATH = DATA_DIR + "/all_epochs_filtered.pkl"

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

all_epochs = {}
all_bads = []

for key_idx, (key, frequency_params) in enumerate(frequencies.items()):
    freq_epochs = []
    for acq_idx, acq_time in enumerate(acq_times):

        with open(DATA_DIR + "/preprocessing_" + acq_time + ".pkl", "rb") as f:
            preprocessing = pickle.load(f)

        raw_filtered = preprocessing.apply_filters(
            preprocessing.raw,
            frequency_params,
            notch_filter=False,
        )
        epochs = preprocessing.create_epochs(raw_filtered)
        freq_epochs.append(epochs)

        all_bads.extend(epochs.info["bads"])

    for acq_idx, epoch in enumerate(freq_epochs):
        epoch.info["bads"] = all_bads

    all_epochs[key] = mne.concatenate_epochs(freq_epochs)

with open(ALL_EPOCHS_PATH, "wb") as f:
    pickle.dump(all_epochs, f)
