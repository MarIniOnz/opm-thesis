"""Creates epochs for each frequency band and acquisition time.

This script creates epochs for each frequency band and acquisition time. The epochs
are saved in a list of lists of lists. The first index corresponds to the frequency
band, the second index corresponds to the acquisition time, and the third index
corresponds to the epoch.
"""
import pickle
import mne

DATA_SAVE = "./data/"

# Frequencies in Hz
low_freq = dict({"l_freq": 0.01, "h_freq": 5})
mid_freq = dict({"l_freq": 5, "h_freq": 120})
all_data = dict({"l_freq": 0.01, "h_freq": 120})
alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
high_gamma = dict({"l_freq": 60, "h_freq": 120})
low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
mid_gamma = dict({"l_freq": 45, "h_freq": 75})
all_gamma = dict({"l_freq": 30, "h_freq": 120})

frequencies = {
    # "low_freq": low_freq,
    "mid_freq": mid_freq,
    # "all_data": all_data,
    #     "alpha": alpha,
    #     "beta": beta,
    #     "low_gamma": low_gamma,
    #     "high_gamma": high_gamma,
    #     "low_mid_gamma": low_mid_gamma,
    #     "mid_gamma": mid_gamma,
    #     "all_gamma": all_gamma,
}


for key_idx, (key, frequency_params) in enumerate(frequencies.items()):
    freq_epochs = []
    all_bads = []

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
            preprocessing.raw_corrected,
            frequency_params,
            notch_filter=False,
        )
        epochs = preprocessing.create_epochs(raw_filtered, gestures=True)
        freq_epochs.append(epochs)
        all_bads.extend(preprocessing.epochs.info["bads"])

    for acq_idx, epoch in enumerate(freq_epochs):
        epoch.info["bads"] = all_bads
        epoch.drop_bad()
        epoch.pick(picks="meg", exclude="bads")

    all_epochs = mne.concatenate_epochs(freq_epochs)
    file_name = DATA_SAVE + "gestures_epochs/freq_bands/" + key + "_all_epochs.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)

    all_epochs.decimate(4)
    file_name = (
        DATA_SAVE + "gestures_epochs/freq_bands/" + key + "_all_epochs_decimated.pkl"
    )

    with open(file_name, "wb") as f:
        pickle.dump(all_epochs, f)
