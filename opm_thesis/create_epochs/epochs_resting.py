"""Creating epochs for resting state data."""
import pickle as pkl
from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing


DATA_DIR = r"/Users/martin.iniguez/Desktop/master_thesis/data_nottingham"
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]
raw, events, events_id = get_data_mne(DATA_DIR, day="20230622", acq_time=acq_times[4])

filter_params = {"method": "iir"}
preprocessing = Preprocessing(raw, events, events_id, filter_params=filter_params)

SAVE_PATH = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed/resting_164054.pkl"
)

with open(SAVE_PATH, "wb") as f:
    pkl.dump(preprocessing, f)

DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
RESTING_PATH = DATA_DIR + "/resting_epochs.pkl"

alpha = dict({"l_freq": 8, "h_freq": 12})
beta = dict({"l_freq": 12, "h_freq": 30})
low_gamma = dict({"l_freq": 30, "h_freq": 60})
high_gamma = dict({"l_freq": 60, "h_freq": 120})

frequencies = {
    "alpha": alpha,
    "beta": beta,
    "low_gamma": low_gamma,
    "high_gamma": high_gamma,
}

resting_epochs = {}
for key, frequency_params in frequencies.items():
    raw_filtered = preprocessing.apply_filters(
        preprocessing.raw,
        frequency_params,
        notch_filter=False,
    )
    resting_epochs[key] = preprocessing.create_epochs(raw_filtered)

with open(RESTING_PATH, "wb") as f:
    pkl.dump(resting_epochs, f)
