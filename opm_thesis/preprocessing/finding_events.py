"""Preprocessing of the data.

This script contains the preprocessing of the data. Takes the acquisition times and
creates a Preprocessing object which contains the raw data, events, and epochs."""
import pickle
import time
from os import mkdir

from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

DATA_DIR = r"/Users/martin.iniguez/Desktop/master-thesis/data_nottingham"

acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]
time_now = time.strftime("%m%d-%H%M")

# Create the folder to save the data
DATA_SAVE = (
    r"/Users/martin.iniguez/Desktop/master-thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed/" + time_now
)
mkdir(DATA_SAVE)

for i in range(4):
    raw, events, event_id = get_data_mne(
        DATA_DIR, day="20230622", acq_time=acq_times[i]
    )
    filter_params = {"method": "iir"}

    preprocessing = Preprocessing(raw, events, event_id, filter_params=filter_params)
    name = DATA_SAVE + "/preprocessing_" + acq_times[i] + ".pkl"
    # Save preprocessing object in data_
    with open(name, "wb") as f:
        pickle.dump(preprocessing, f)
