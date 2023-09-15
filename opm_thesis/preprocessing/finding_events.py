"""Preprocessing of the data.

This script contains the preprocessing of the data. Takes the acquisition times and
creates a Preprocessing object which contains the raw data, events, and epochs."""
import pickle
from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

DATA_DIR = r"/Users/martin.iniguez/Desktop/master_thesis/data_nottingham"
DATA_SAVE = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]

for i in range(4):
    raw, events, event_id = get_data_mne(
        DATA_DIR, day="20230622", acq_time=acq_times[i]
    )

    preprocessing = Preprocessing(raw, events, event_id)
    name = DATA_SAVE + "/preprocessing_" + acq_times[i] + ".pkl"
    # Save preprocessing object in data_
    with open(name, "wb") as f:
        pickle.dump(preprocessing, f)
