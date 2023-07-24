from opm_thesis.read_files.cMEG2fif_bespoke import get_data_mne
from opm_thesis.preprocessing.preprocessing import Preprocessing

data_dir = r"C:\Users\user\Desktop\MasterThesis\data_nottingham"
acq_times = ["155445", "160513", "161344", "163001", "164054", "165308"]
acq_times = ["161344"]

for acq_time in acq_times:
    raw, events, events_id = get_data_mne(data_dir, day="20230622", acq_time=acq_time)

wrong_idx = []
new_events = events.copy()
for i in range(len(events)-1):
    if events[i][2] <= 5:
      # Look at our events from a number under 5 to 7 (more than one meaning it is wrong)
      if events[i+1][2] != 2**(events[i][2]+2):
        wrong_idx.append(i)
        if events[i+1][2] != 7:
            events[i+1][2] = 9
            # Add this values to the events_id

