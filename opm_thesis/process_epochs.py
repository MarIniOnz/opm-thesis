import mne
import pickle


data_save = (
    r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed"
)


def save_epochs():
    all_bads = []
    acq_times = ["155445", "160513", "161344", "163001"]

    alpha = dict({"l_freq": 8, "h_freq": 12})
    beta = dict({"l_freq": 12.5, "h_freq": 30})
    low_gamma = dict({"l_freq": 30, "h_freq": 60})
    low_mid_gamma = dict({"l_freq": 30, "h_freq": 45})
    mid_gamma = dict({"l_freq": 45, "h_freq": 75})
    high_gamma = dict({"l_freq": 60, "h_freq": 120})
    all_gamma = dict({"l_freq": 30, "h_freq": 120})

    frequencies = {
        "alpha": alpha,
        "beta": beta,
        "low_gamma": low_gamma,
        "low_mid_gamma": low_mid_gamma,
        "mid_gamma": mid_gamma,
        "high_gamma": high_gamma,
        "all_gamma": all_gamma,
    }
    for key, frequency_params in frequencies.items():
        preprocessed_epochs = list()
        for i in range(4):
            preprocessing = pickle.load(
                open(data_save + "\\preprocessing_" + acq_times[i] + ".pkl", "rb")
            )
            raw_filtered = preprocessing.apply_filters(
                preprocessing.raw, frequency_params
            )
            picks = mne.pick_types(raw_filtered.info, meg="mag", exclude="bads")
            hilbert_transformed = raw_filtered.copy().apply_hilbert(picks=picks)
            hilbert_epochs = preprocessing.create_epochs(hilbert_transformed)

            preprocessed_epochs.append(hilbert_epochs)
            all_bads.extend(hilbert_epochs.info["bads"])

        for i in range(len(preprocessed_epochs)):
            preprocessed_epochs[i].info["bads"] = all_bads

        all_epochs = mne.concatenate_epochs(preprocessed_epochs)
        name = data_save + "\\epochs\\hilbert_" + key + "_all_epochs.pkl"

        with open(name, "wb") as f:
            pickle.dump(all_epochs, f)


if "main" in __name__:

    save_epochs()

# from autoreject import AutoReject
# Use 'n_jobs=-1' to use all available CPU cores for parallel processing
# ar = AutoReject(n_jobs=-1, verbose='tqdm')
# Fit the autoreject object to the data (only MEG channels)
# ar.fit(epochs.copy().pick(['meg']))
# clean_epochs, reject_log = ar.transform(epochs.copy().pick(['meg']), return_log=True)
# print(ar)
