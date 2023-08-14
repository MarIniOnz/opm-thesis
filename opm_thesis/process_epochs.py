import mne
import pickle


data_save = (
    r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
)
# If my file exists in the data folder, load it. Otherwise, create it.
try:
    with open(data_save + "\\all_epochs.pkl", "rb") as f:
        all_epochs = pickle.load(f)

except FileNotFoundError:

    def save_epochs():
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
        name = data_save + "\\all_epochs.pkl"
        # Save preprocessing object in data_
        with open(name, "wb") as f:
            pickle.dump(all_epochs, f)

        return all_epochs

    all_epochs = save_epochs()

all_epochs


# from autoreject import AutoReject
# Use 'n_jobs=-1' to use all available CPU cores for parallel processing
# ar = AutoReject(n_jobs=-1, verbose='tqdm')
 # Fit the autoreject object to the data (only MEG channels)
# ar.fit(epochs.copy().pick(['meg']))
# clean_epochs, reject_log = ar.transform(epochs.copy().pick(['meg']), return_log=True)
# print(ar)
