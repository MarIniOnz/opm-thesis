"""Preprocessing of the data.

This script contains the preprocessing of the data. Takes the acquisition times and
creates a Preprocessing object which contains the raw data, events, and epochs."""
import pickle
import time
import glob
import os

import mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

DATA_DIR = "./data/gestures/"
raw_files = glob.glob(DATA_DIR + "*-raw.fif.gz")
time_now = time.strftime("%m%d-%H%M")

# Create the folder to save the data
DATA_SAVE = DATA_DIR + "preprocessed" + time_now

if not os.path.exists(DATA_SAVE):
    os.mkdir(DATA_SAVE)

for file in raw_files:
    # Load the raw data
    raw = mne.io.read_raw_fif(file, preload=True)
    file_name = file.split("/")[-1]

    # Load the epochs
    epochs_filename = file_name.replace("-raw.fif.gz", "-epo.fif.gz")
    epochs = mne.read_epochs(DATA_DIR + epochs_filename, preload=True)

    # Find events
    events = epochs.events
    event_id = epochs.event_id
    filter_params = {"method": "iir"}

    preprocessing = Preprocessing(
        raw,
        events,
        event_id,
        gestures=True,
        filter_params=filter_params,
    )
    name = DATA_SAVE + "/preprocessing_gestures_" + file_name[-12] + ".pkl"

    # Save preprocessing object
    with open(name, "wb") as f:
        pickle.dump(preprocessing, f)
