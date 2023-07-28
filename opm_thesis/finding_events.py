from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

data_dir = r"C:\Users\user\Desktop\MasterThesis\data_nottingham"
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]
raw, events, events_id = get_data_mne(data_dir, day="20230622", acq_time=acq_times[3])

preprocessing = Preprocessing(raw, events, events_id)
print(preprocessing.wrong_trials)
