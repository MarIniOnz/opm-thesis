import pickle
import mne
import numpy as np
import matplotlib as mpl
from mne.time_frequency import tfr_morlet


data_save = (
    r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
)
preprocessed_epochs = list()
all_bads = []
acq_times = ["155445", "160513", "161344", "163001"]

for i in range(4):
    preprocessing = pickle.load(
        open(data_save + "\\preprocessing_" + acq_times[i] + ".pkl", "rb")
    )
    preprocessed_epochs.append(preprocessing.epochs)
    all_bads.extend(preprocessing.epochs.info["bads"])

for i in range(4):
    preprocessed_epochs[i].info["bads"] = all_bads

all_epochs = mne.concatenate_epochs(preprocessed_epochs)
all_epochs


mpl.use("TkAgg")

freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.0  # different number of cycle per frequency
power = tfr_morlet(
    all_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    use_fft=True,
    return_itc=False,
    decim=2,
    n_jobs=None,
)
##
power.plot_topo(
    baseline=(-1, -0.5), tmin=-1, tmax=1, mode="zlogratio", fig_facecolor="w"
)
