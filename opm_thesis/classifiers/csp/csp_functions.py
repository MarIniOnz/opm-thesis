"Functions to use for CSP classification"
import warnings
from typing import List, Tuple
import pickle

from math import exp, log
import mne
from mne.viz.topomap import _find_topomap_coords
from mne.decoding import CSP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from opm_thesis.preprocessing.utils import get_closest_sensors

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")


def csp_classify(
    id_pair: List[int],
    freq: str,
    data_dir: str,
    num: int = None,
    dim: str = "all",
    n_components: int = 4,
    normalize_data: bool = True,
) -> float:
    """Classify the given ID pair using CSP

    :param id_pair: The ID pair to classify
    :type id_pair: List[int]
    :param freq: The frequency band to use
    :type freq: str
    :param data_dir: The directory where the data is stored
    :type data_dir: str
    :param n_components: The number of CSP components to use, defaults to 4
    :type n_components: int, optional
    :param normalize_data: Whether to normalize the data, defaults to True
    :type normalize_data: bool, optional
    :return: The classification score
    :rtype: float
    """
    with open(data_dir + f"{freq}_all_epochs.pkl", "rb") as f:
        epochs = pickle.load(f)
        epochs = epochs.decimate(2)

    data, labels, _ = process_data(
        epochs, id_pair, dim=dim, num=num, normalize_data=normalize_data
    )

    clf = LinearDiscriminantAnalysis()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    scores = []
    mapping = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5}
    id_pair
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
    id_pair: List[int],
    freq: str,
    data_dir: str,
    num: int = None,
    dim: str = "[X]",
    n_components: int = 4,
    normalize_data: bool = True,
) -> None:
    """Plot the CSP patterns for the given ID pair

    :param id_pair: The ID pair to plot
    :type id_pair: List[int]
    :param freq: The frequency band to use
    :type freq: str
    :param data_dir: The directory where the data is stored
    :type data_dir: str
    :param dim: The dimension to use, defaults to "[X]"
    :type dim: str, optional
    :param n_components: The number of CSP components to use, defaults to 4
    :type n_components: int, optional
    :param normalize_data: Whether to normalize the data, defaults to True
    :type normalize_data: bool, optional
    """
    # Load the data
    with open(f"{data_dir}/{freq}_all_epochs.pkl", "rb") as f:
        epochs = pickle.load(f)
        epochs = epochs.decimate(2)

    # Select epochs for the given ID pair
    data, labels, info = process_data(epochs, id_pair, dim, num, normalize_data)

    # Fit CSP
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit(data, labels)

    # Plot CSP patterns
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 5))

    pos = _find_topomap_coords(info, picks="LQ" + dim, sphere=None)[0]

    for i, pattern in enumerate(csp.patterns_.T):
        if i >= n_components:  # Only plot the desired number of components
            break
        if n_components == 1:
            mne.viz.plot_topomap(
                pattern,
                info,
                show=False,
                axes=axes,
                ch_type="mag",
            )
            axes.scatter(
                *pos, color="black", s=100, label="LQ" + dim
            )  # Highlight with a red dot
        else:
            mne.viz.plot_topomap(
                pattern,
                info,
                axes=axes[i],
                show=False,
                ch_type="mag",
            )
            axes[i].scatter(
                *pos, color="red", s=100, label="LQ" + dim
            )  # Highlight with a red dot
            axes[i].set_title(f"Component {i+1}")

    plt.tight_layout()
    # Set global title for the figure
    fig.suptitle(
        f"Frequency: {freq}, ID Pair: {id_pair}, Dimension: {dim[1]}",
        fontsize=16,
        y=1.02,
    )


def process_data(
    epochs: mne.Epochs, id_pair: List[int], dim="all", num=None, normalize_data=True
) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
    """Process the data for the given ID pair

    :param epochs: The epochs to process
    :type epochs: mne.Epochs
    :param id_pair: The ID pair to process
    :type id_pair: List[int]
    :param dim: The dimension to use, defaults to "all"
    :type dim: str, optional
    :param normalize_data: Whether to normalize the data, defaults to True
    :type normalize_data: bool, optional
    :return: The processed data, labels and info
    :rtype: Tuple[np.ndarray, np.ndarray, mne.Info]
    """
    indices = np.where(
        np.logical_or(
            epochs.events[:, -1] == id_pair[0],
            epochs.events[:, -1] == id_pair[1],
        )
    )[0]
    pair_epochs = epochs[indices].pick("meg", exclude="bads")

    if num is not None:
        channel_names = get_closest_sensors(epochs.info, "LQ[X]", num)
    else:
        channel_names = pair_epochs.ch_names

    # Select only the desired dimension
    if "all" not in dim:
        channel_names = [ch for ch in channel_names if dim in ch]

    pair_epochs = pair_epochs.pick(channel_names)

    data_epochs = pair_epochs.get_data()
    labels = pair_epochs.events[:, -1]
    info = pair_epochs.info

    if normalize_data:
        mean = data_epochs.mean(axis=0, keepdims=True)
        std = data_epochs.std(axis=0, keepdims=True)
        data_epochs = (data_epochs - mean) / std

    return data_epochs, labels, info
