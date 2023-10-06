"""Preprocessing of the data.

This script contains the preprocessing of the data. Takes the acquisition times and
creates a Preprocessing object which contains the raw data, events, and epochs."""
from typing import Tuple
import mne
import numpy as np
from mne.io import RawArray

from opm_thesis.preprocessing.utils import (
    get_closest_sensors,
    create_fixed_length_events,
    detect_bad_channels_by_zscore,
    create_resting_epochs,
)


class Preprocessing:
    """Preprocessing of the data.

    This class contains the preprocessing of the data. That includes:
    - Filtering
    - Epoching
    - Artifact removal
    - Signal space separation
    - Normalization
    """

    def __init__(
        self,
        raw: RawArray,
        events: np.ndarray,
        event_id: dict,
        filter_params: dict = {},
        notch_filter: bool = True,
        epochs_params: dict = {},
        signal_sep_params: dict = {},
        artifact_params: dict = {},
        channels_params: dict = {},
    ) -> None:
        self.raw = raw
        assert "sfreq" in raw.info, "Sampling frequency not found in raw.info"
        self.samp_freq = raw.info["sfreq"]

        self.raw = self.apply_filters(
            self.raw, filter_params=filter_params, notch_filter=notch_filter
        )

        self.events, self.event_id, self.wrong_trials = self.preprocess_events(
            events, event_id
        )
        self.epochs = self.create_epochs(self.raw, **epochs_params)

        artifact_trials = self.remove_artifacts(artifact_params=artifact_params)
        self.wrong_trials.extend(artifact_trials)

        self.raw_reduced, self.epochs_reduced = self.select_channels(
            self.raw, channel_params=channels_params
        )

        # self.data = self.signal_space_separation(signal_sep_params)
        # self.data = self.normalize_data()

    def apply_filters(self, raw: RawArray, filter_params: dict, notch_filter=True):
        """Apply filters to raw data.

        :param raw: Raw data
        :type raw: RawArray
        :param filter_params: Parameters for filtering
        :type filter_params: dict
        :param notch_filter: Whether to apply notch filter, defaults to True
        :type notch_filter: bool, optional
        :return: Filtered raw data
        :rtype: RawArray
        """
        default_params = dict(
            {"l_freq": 0.01, "h_freq": 330, "fir_design": "firwin", "phase": "zero"}
        )
        default_params.update(filter_params)

        notch_freq = 50
        if notch_filter:
            raw = raw.copy().notch_filter(freqs=notch_freq, fir_design="firwin")

        return raw.copy().filter(**default_params)

    def preprocess_events(
        self, events: np.ndarray, event_id: dict
    ) -> Tuple[np.ndarray, dict, list]:
        """Preprocess events array.

        Marks trials with wrong button presses or no button pressed.

        :param events: Events array (3 columns: sample idx, duration, event code)
        :type events: np.ndarray
        :param events_id: Dictionary with event codes as keys and event names as values
        :type events_id: dict
        :param wrong_previous_trial: List of indexes of trials that went wrong (indices
            corresponding to the end of that trial)
        :type wrong_previous_trial: list
        :return: Preprocessed events array, updated event_id dictionary
        :rtype: Tuple[np.ndarray, dict]
        """
        if events.size == 0:
            events, events_id = create_fixed_length_events(self.raw, 2.0)
            return events, events_id, []

        # Get indexes of events == 7
        idx_trials = np.where(events[:, 2] == 7)[0]
        wrong_previous_trial = []

        for i, trial_idx in enumerate(idx_trials):
            # Get number of events in previous trial
            num_events_trial = (
                trial_idx
                if trial_idx == idx_trials[0]
                else trial_idx - idx_trials[i - 1]
            )
            wrong_previous_trial.append(i)

            if num_events_trial == 2:
                # Button not pressed
                events[trial_idx - 1, 2] = 6
            elif num_events_trial == 3 and events[trial_idx - 1, 2] != 2 ** (
                events[trial_idx - 2, 2] + 2
            ):
                # Wrong button pressed
                events[trial_idx - 2, 2] = 9
                events[trial_idx - 1, 2] = 11
            elif num_events_trial > 3:
                # Too many buttons pressed
                events[trial_idx - num_events_trial + 1 : trial_idx, 2] = 12
                events[trial_idx - num_events_trial + 1, 2] = 10
            else:
                # Correct button pressed
                wrong_previous_trial.pop(-1)

        event_id["cue_not_answered"] = 6
        event_id["cue_wrong_button"] = 9
        event_id["cue_multiple_buttons"] = 10
        event_id["press_wrong_button"] = 11
        event_id["press_multiple_buttons"] = 12

        return events, event_id, wrong_previous_trial

    def create_epochs(self, raw: RawArray, **epochs_params):
        """Create epochs from raw data.

        :param raw: Raw data
        :type raw: RawArray
        :param epochs_params: Parameters for epoch creation
        :type epochs_params: dict
        :return: Epochs object
        :rtype: mne.Epochs
        """
        if 999 in self.event_id.values():
            return mne.Epochs(
                raw,
                self.events,
                self.event_id,
                tmin=0,
                tmax=2.0,
                baseline=None,
                detrend=1,
                preload=True,
            )

        # Setting default parameters
        default_params = {}
        cue_ids = np.arange(1, 6)
        default_params["preload"] = True
        default_params["event_id"] = [2 ** (i + 2) for i in cue_ids]
        default_params["tmin"], default_params["tmax"] = (-2.0, 2.0)

        time_calculation = None
        if time_calculation == "avg":
            default_params["tmin"], default_params["tmax"] = self.calculate_avg_times(
                cue_ids=cue_ids
            )
        elif time_calculation == "max":
            default_params["tmin"], default_params["tmax"] = self.calculate_max_times(
                cue_ids=cue_ids
            )

        default_params.update(epochs_params)

        event_id_interest = {
            name: code
            for name, code in self.event_id.items()
            if code in default_params["event_id"]
        }
        default_params["event_id"] = event_id_interest

        return mne.Epochs(
            raw=raw,
            events=self.events,
            **default_params,
        )

    def select_channels(
        self, raw: RawArray, channel_params: dict = None
    ) -> Tuple[RawArray, mne.Epochs]:
        """Select the channels to use for the analysis.

        :param raw: Raw data
        :type raw: RawArray
        :param channel_params: Parameters for selecting channels
        :type channel_params: dict
        :return: Raw data and epochs with reduced number of channels
        :rtype: Tuple[RawArray, mne.Epochs]
        """
        default_params = {
            "centre_channel": "LQ[X]",
            "num_channels": 27,
            "zscore_threshold": 1.5,
        }
        default_params.update(channel_params)

        # Get the closest sensors to the centre channel
        closest_sensors_names = get_closest_sensors(
            raw.info,
            centre_channel=default_params["centre_channel"],
            num_channels=default_params["num_channels"],
        )
        raw_reduced = raw.copy().pick(closest_sensors_names)

        bad_channel_names_X = detect_bad_channels_by_zscore(
            raw_reduced,
            coordinate="X",
            zscore_threshold=default_params["zscore_threshold"],
        )
        bad_channel_names_Y = detect_bad_channels_by_zscore(
            raw_reduced,
            coordinate="Y",
            zscore_threshold=default_params["zscore_threshold"],
        )
        bad_channel_names_Z = detect_bad_channels_by_zscore(
            raw_reduced,
            coordinate="Z",
            zscore_threshold=default_params["zscore_threshold"],
        )
        closest_sensors_names = np.array(
            [
                name
                for name in closest_sensors_names
                if name not in bad_channel_names_X
                and name not in bad_channel_names_Y
                and name not in bad_channel_names_Z
            ]
        )

        # Create a new raw object with the reduced number of channels and epochs
        raw_reduced = raw.copy().pick(closest_sensors_names)
        epochs_reduced = self.epochs.copy().pick(closest_sensors_names)

        return raw_reduced, epochs_reduced

    def signal_space_separation(self, signal_sep_params: dict = None):
        """Apply signal space separation to the data."""
        pass

    def remove_artifacts(
        self, remove_type: str = "AutoReject", artifact_params: dict = None
    ):
        """
        Remove artifacts from the data.

        :param remove_type: Type of artifact removal. Default: "AutoReject"
            Options: "AutoReject", "MNE_Manual", "VarianceThreshold"
        :type remove_type: str, optional
        """

        # from autoreject import AutoReject
        # Use 'n_jobs=-1' to use all available CPU cores for parallel processing
        # ar = AutoReject(n_jobs=-1, verbose='tqdm')
        # Fit the autoreject object to the data (only MEG channels)
        # ar.fit(epochs.copy().pick(['meg']))
        # clean_epochs, reject_log = ar.transform(epochs.copy().pick(['meg']),
        #      return_log=True)
        # print(ar)

        artifact_trials = []

        return artifact_trials

    def normalize_data(self):
        """Normalize the data."""
        pass
