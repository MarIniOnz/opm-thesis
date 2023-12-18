"""Saving all epochs of an analyzed preprocessing object."""
import pickle
import numpy as np

DATA_DIR = "/Users/martin.iniguez/Desktop/master-thesis/opm-thesis/data/"
acq_times = ["155445", "160513", "161344", "163001"]

all_bads = []
preprocessed_epochs = []
baseline = []

for i in range(4):
    with open(
        DATA_DIR
        + "data_nottingham_preprocessed/analyzed/preprocessing_"
        + acq_times[i]
        + ".pkl",
        "rb",
    ) as f:
        preprocessing = pickle.load(f)

    epochs_corrected = preprocessing.epochs_corrected
    preprocessed_epochs.append(epochs_corrected)

    all_bads.extend(epochs_corrected.info["bads"])
    baseline.append(epochs_corrected.baseline[-1])

baseline = np.mean(baseline)
for i in range(4):
    preprocessed_epochs[i].apply_baseline(baseline=(-2, baseline))
    preprocessed_epochs[i].info["bads"] = all_bads
    preprocessed_epochs[i].drop_bad()

    with open(DATA_DIR + "epochs/corrected_epochs" + acq_times[i] + ".pkl", "wb") as f:
        pickle.dump(preprocessed_epochs[i], f)
