import pickle
from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

data_dir = r"C:\Users\user\Desktop\MasterThesis\data_nottingham"
data_save = r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]

for i in range(4):
    raw, events, event_id = get_data_mne(data_dir, day="20230622", acq_time=acq_times[i])

    preprocessing = Preprocessing(raw, events, event_id)
    name  = data_save + "\\preprocessing_" + acq_times[i] + ".pkl"
    # Save preprocessing object in data_
    with open(name, "wb") as f:
        pickle.dump(preprocessing, f)
