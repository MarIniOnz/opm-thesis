"""Creating epochs for resting state data."""

import pickle as pkl
from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing


DATA_DIR = "../data_nottingham/"
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]
raw, events, events_id = get_data_mne(DATA_DIR, day="20230622", acq_time=acq_times[4])

preprocessing = Preprocessing(raw, events, events_id)

DATA_DIR = "./data/"
SAVE_PATH = DATA_DIR + "resting/resting_164054.pkl"

with open(SAVE_PATH, "wb") as f:
    pkl.dump(preprocessing, f)

RESTING_PATH = DATA_DIR + "resting_epochs/"

all_data = dict({"l_freq": 0.01, "h_freq": 120})
alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
gamma = dict({"l_freq": 30, "h_freq": 60})

frequencies = {
    "all_data": all_data,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
}

resting_epochs = {}
for key, frequency_params in frequencies.items():
    raw_filtered = preprocessing.apply_filters(
        preprocessing.raw,
        frequency_params,
        notch_filter=False,
    )
    resting_epochs = preprocessing.create_epochs(raw_filtered)

    FILENAME = RESTING_PATH + key + "_all_epochs.pkl"

    with open(FILENAME, "wb") as f:
        pkl.dump(resting_epochs, f)
