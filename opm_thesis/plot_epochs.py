# import mne
import numpy as np
from mne.time_frequency import tfr_morlet

## load epochs here
data_dir = r"D:\PhD\data\2023-06-21_nottingham"


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.0  # different number of cycle per frequency
power = tfr_morlet(
    epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    use_fft=True,
    return_itc=False,
    decim=2,
    n_jobs=None,
)
##
power.plot_topo(baseline=(-1, -0.5), tmin=-1, tmax=1, mode='zlogratio', fig_facecolor='w')
