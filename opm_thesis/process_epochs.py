import mne
import pickle

data_save = r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
preprocessed_epochs = list()
all_bads = []
acq_times = ["155445", "160513", "161344", "163001"]

for i in range(4):
    preprocessing = pickle.load(open(data_save + "\\preprocessing_" + acq_times[i] + ".pkl", "rb"))
    preprocessed_epochs.append(preprocessing.epochs)
    all_bads.extend(preprocessing.epochs.info["bads"])

for i in range(4):
    preprocessed_epochs[i].info["bads"] = all_bads

all_epochs = mne.concatenate_epochs(preprocessed_epochs)
all_epochs
# from autoreject import AutoReject
# ar = AutoReject(n_jobs=-1, verbose='tqdm')  # Use 'n_jobs=-1' to use all available CPU cores for parallel processing
# ar.fit(epochs.copy().pick(['meg']))  # Fit the autoreject object to the data (only MEG channels
# clean_epochs, reject_log = ar.transform(epochs.copy().pick(['meg']), return_log=True)
# print(ar)
