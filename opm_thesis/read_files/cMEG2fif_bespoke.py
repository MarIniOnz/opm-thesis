# -*- coding: utf-8 -*-
"""
Load OPM Data, convert for use in MNE

@author: Ryan Hill & Molly Rea
"""
#%% Import packages
import numpy as np
import mne
import os
import pandas as pd
import json
import tkinter as tk
from tkinter import filedialog

# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mne.io.constants import FIFF

from opm_thesis.read_files.utils import (
    read_old_cMEG,
    find_matching_indices,
    create_chans_dict,
    get_channels_and_data,
    calc_pos,
)

mne.viz.set_3d_backend("pyvistaqt")

#%% configure subjects directory
subjects_dir = ""
subject = "11766"

root = tk.Tk()  # create a Tk root window (GUI window)
root.withdraw()  # hide the Tk root window

#%% Data filename and path
file_path = filedialog.askopenfilename(title="Choose cMEG File")
file_path_split = os.path.split(file_path)
fpath = file_path_split[0] + "/"
fname = file_path_split[1]

#%% Load Data
# Requires a single cMEG file, doesn't concatenate runs yet
print("Loading File")

# Load data
data_input = read_old_cMEG(fpath + fname)
time = data_input[0, :]
data_raw = data_input[1:, :]
del data_input

fname_pre = fname.split("_meg.cMEG")[0]
f = open(fpath + fname_pre + "_meg.json")
tsv_file = {
    "channels": pd.read_csv(fpath + fname_pre + "_channels.tsv", sep="\t"),
    "HelmConfig": pd.read_csv(fpath + fname_pre + "_HelmConfig.tsv", sep="\t"),
    "SensorTransform": pd.read_csv(
        fpath + fname_pre + "_SensorTransform.tsv", header=None, sep="\t"
    ),
    "JSON": json.load(f),
}
f.close()
samp_freq = tsv_file["JSON"]["SamplingFrequency"]

#%% Sensor indexes and locations
names = tsv_file["channels"]["name"]
sensors = tsv_file["HelmConfig"]["Sensor"]
loc_idx = find_matching_indices(names, sensors)
chans = create_chans_dict(tsv_file, loc_idx)

#%% Sensor information
print("Sorting Sensor Information")
try:
    ch_scale = pd.Series.tolist(tsv_file["channels"]["nT/V"])
except KeyError:
    tsv_file["channels"].rename(columns={"nT0x2FV": "nT/V"}, inplace=True)
    ch_scale = pd.Series.tolist(tsv_file["channels"]["nT/V"])
ch_names, ch_types, data = get_channels_and_data(data_raw, tsv_file, ch_scale)
sfreq = samp_freq

#%% Create MNE info object
print("Creating MNE Info")
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
info["line_freq"] = tsv_file["JSON"]["PowerLineFrequency"]

#%% Sort sensor locations
print("Sensor Location Information")
nmeg = nstim = nref = 0
chs = list()

for ii in range(tsv_file["channels"].shape[0]):
    # Create channel information
    ch = dict(
        scanno=ii + 1,
        range=1.0,
        cal=1.0,
        loc=np.full(12, np.nan),
        unit_mul=FIFF.FIFF_UNITM_NONE,
        ch_name=tsv_file["channels"]["name"][ii].replace(" ", ""),
        coil_type=FIFF.FIFFV_COIL_NONE,
    )

    pos = chans["Locations"][ii]
    ori = chans["Orientations"][ii]

    # Calculate sensor position
    if sum(np.isnan(pos)) == 0:
        ch["loc"] = calc_pos(pos, ori)

    # Update channel depending on type
    if chans["Channel_Type"][ii].replace(" ", "") == "TRIG":  # its a trigger!
        nstim += 1
        info["chs"][ii].update(
            logno=nstim,
            coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
            kind=FIFF.FIFFV_STIM_CH,
            unit=FIFF.FIFF_UNIT_V,
            cal=1,
        )

    elif chans["Channel_Type"][ii].replace(" ", "") == "MISC":  # its a BNC channel
        nref += 1
        info["chs"][ii].update(
            logno=nstim,
            coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
            kind=FIFF.FIFFV_STIM_CH,
            unit=FIFF.FIFF_UNIT_V,
            cal=1,
        )

    elif sum(np.isnan(pos)) == 3:  # its a sensor with no location info
        nref += 1
        info["chs"][ii].update(
            logno=nref,
            coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
            kind=FIFF.FIFFV_REF_MEG_CH,
            unit=FIFF.FIFF_UNIT_T,
            coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
            cal=1e-9 / tsv_file["channels"]["nT/V"][ii],
        )

    else:  # its a sensor!
        nmeg += 1
        info["chs"][ii].update(
            logno=nmeg,
            coord_frame=FIFF.FIFFV_COORD_DEVICE,
            kind=FIFF.FIFFV_MEG_CH,
            unit=FIFF.FIFF_UNIT_T,
            coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
            loc=ch["loc"],
            cal=1e-9 / tsv_file["channels"]["nT/V"][ii],
        )

    chs.append(ch)

info["dev_head_t"] = mne.transforms.Transform(
    "meg", "head", pd.DataFrame(tsv_file["SensorTransform"]).to_numpy()
)


#%% Create MNE raw object
print("Create raw object")
raw = mne.io.RawArray(data, info)

# Set bad channels defined in channels.tsv
idx = tsv_file["channels"].status.str.strip() == "Bad"
bad_ch = tsv_file["channels"].name[idx.values]
raw.info["bads"] = bad_ch.str.replace(" ", "").to_list()

#%% Create events
stm_misc_chans = mne.pick_types(info, stim=True, misc=True)
trig_data = 1 * np.array(data[stm_misc_chans, :] > 2)
trig_ID, on_inds = np.where(np.diff(trig_data, axis=1) == 1)
if len(trig_ID) > 0:
    events = np.concatenate(
        [
            np.expand_dims(on_inds, axis=1) + 1,
            np.expand_dims(np.zeros(np.shape(on_inds)), axis=1),
            np.expand_dims(trig_ID, axis=1) + 1,
        ],
        axis=1,
    ).astype(np.int64)

#%% Digitisation and montage

print("Digitisation")
ch_pos = dict()
for ii in range(tsv_file["channels"].shape[0]):
    pos1 = chans["Locations"][ii]
    if sum(np.isnan(pos1)) == 0:
        ch_pos[chans["Channel_Name"][ii].replace(" ", "")] = pos1

mtg = mne.channels.make_dig_montage(ch_pos=ch_pos)
raw.set_montage(mtg)


