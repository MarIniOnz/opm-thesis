import warnings
import pickle
import mne
from typing import List, Tuple
from mne.decoding import CSP

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")


def csp_classify(
    id_pair,
    freq,
    data_dir,
    n_components: int = 4,
    USE_HALF: bool = False,
    NORMALIZE_DATA: bool = True,
):
    with open(data_dir + f"{freq}_all_epochs_decimated.pkl", "rb") as f:
        epochs = pickle.load(f)

    data, labels, _ = process_data(epochs, id_pair, NORMALIZE_DATA)

    clf = LinearDiscriminantAnalysis()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    scores = []
    if id_pair[0] > 7:
        id_pair = [int(id / 8) for id in id_pair]
    print("Computing scores for", id_pair, "in", freq, "band...", end=" ")

    for train_idx, test_idx in cv.split(data):
        train_data_csp = []
        test_data_csp = []

        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]

        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False).fit(
            X_train, y_train
        )
        train_data_csp.append(csp.transform(X_train))
        test_data_csp.append(csp.transform(X_test))

        X_train_csp = np.hstack(train_data_csp)
        X_test_csp = np.hstack(test_data_csp)

        clf.fit(X_train_csp, y_train)
        score = clf.score(X_test_csp, y_test)
        scores.append(score)

    score = np.mean(scores) * 100
    print(f"Done. Average Score: {score:.2f}%")
    return score


def plot_csp_patterns(
    id_pair,
    key,
    data_dir,
    dim="X",
    n_components=4,
    NORMALIZE_DATA=True,
):
    # Load the data
    with open(f"{data_dir}/{key}_all_epochs_decimated.pkl", "rb") as f:
        epochs = pickle.load(f)

    # Select epochs for the given ID pair
    data, labels, info = process_data(epochs, id_pair, dim, NORMALIZE_DATA)

    # Fit CSP
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit(data, labels)

    # Plot CSP patterns
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 5))

    for i, pattern in enumerate(csp.patterns_.T):
        if i >= n_components:  # Only plot the desired number of components
            break
        mne.viz.plot_topomap(
            pattern,
            info,
            axes=axes[i],
            show=False,
            ch_type="mag",
        )

        axes[i].set_title(f"Component {i+1}")

    plt.tight_layout()
    # Set global title for the figure
    fig.suptitle(f"Frequency: {key}, ID Pair: {id_pair}", fontsize=16, y=1.02)


def process_data(
    epochs: mne.Epochs, id_pair: List[int], dim="all", NORMALIZE_DATA=True
) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    indices = np.where(
        np.logical_or(
            epochs.events[:, -1] == id_pair[0],
            epochs.events[:, -1] == id_pair[1],
        )
    )[0]
    pair_epochs = epochs[indices].pick("meg", exclude="bads")

    # Select only the desired dimension
    if dim != "all":
        channel_names = [ch for ch in pair_epochs.ch_names if dim in ch]
        pair_epochs = pair_epochs.pick(channel_names)

    data_epochs = pair_epochs.get_data()
    labels = pair_epochs.events[:, -1]
    info = pair_epochs.info

    if NORMALIZE_DATA:
        mean = data_epochs.mean(axis=0, keepdims=True)
        std = data_epochs.std(axis=0, keepdims=True)
        data_epochs = (data_epochs - mean) / std

    return data_epochs, labels, info
