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
        gestures: bool = False,
        filter_params: dict = {},
        notch_filter: bool = True,
        epochs_params: dict = {},
        channels_params: dict = {},
    ) -> None:
        """Initialize the preprocessing object. Runs the preprocessing pipeline,
        which includes:
        - Filtering
        - Epoching
        - Artifact removal
        - Signal space separation
        - Normalization

        :param raw: Raw data
        :type raw: RawArray
        :param events: Events array
        :type events: np.ndarray
        :param event_id: Dictionary with event codes as keys and event names as values
        :type event_id: dict
        :param gestures: Whether the data contains gestures or digits' data.
            False: digits' data, True: gestures' data, defaults to False
        :type gestures: bool, optional
        :param filter_params: Parameters for filtering, defaults to {}
        :type filter_params: dict, optional
        :param notch_filter: Whether to apply notch filter, defaults to True
        :type notch_filter: bool, optional
        :param epochs_params: Parameters for epoch creation, defaults to {}
        :type epochs_params: dict, optional
        :param channels_params: Parameters for selecting channels, defaults to {}
        :type channels_params: dict, optional
        :param signal_sep_params: Parameters for signal space separation, defaults to {}
        :type signal_sep_params: dict, optional
        """

        self.raw = raw
        assert "sfreq" in raw.info, "Sampling frequency not found in raw.info"
        self.samp_freq = raw.info["sfreq"]

        self.raw = self.apply_filters(
            self.raw, filter_params=filter_params, notch_filter=notch_filter
        )

        self.events, self.event_id, self.wrong_trials = self.preprocess_events(
            events, event_id, gestures
        )
        self.epochs = self.create_epochs(self.raw, epochs_params, gestures)

        self.channel_names = self.select_channels(
            self.raw, channel_params=channels_params
        )

        self.raw_corrected, self.epochs_corrected = self.manual_artifact_removal(
            raw=self.raw,
            channel_names=self.channel_names,
            events=self.events,
            event_id=self.event_id,
            epochs_params=epochs_params,
            gestures=gestures,
        )

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
        default_params = dict({"l_freq": 0.01, "h_freq": 120, "phase": "zero"})
        default_params.update(filter_params)

        notch_freq = 50
        if notch_filter:
            raw = raw.copy().notch_filter(freqs=notch_freq, method="iir")

        return raw.copy().filter(**default_params)

    def preprocess_events(
        self, events: np.ndarray, event_id: dict, gestures: bool = False
    ) -> Tuple[np.ndarray, dict, list]:
        """Preprocess events array.

        Marks trials with wrong button presses or no button pressed.

        :param events: Events array (3 columns: sample idx, duration, event code)
        :type events: np.ndarray
        :param events_id: Dictionary with event codes as keys and event names as values
        :type events_id: dict
        :param gestures: Whether the data contains gestures or digits' data.
            False: digits' data, True: gestures' data, defaults to False
        :type gestures: bool, optional
        :return: Preprocessed events array, updated event_id dictionary
        :rtype: Tuple[np.ndarray, dict]
        """
        if gestures:
            return events, event_id, []
        else:
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

    def create_epochs(self, raw: RawArray, epochs_params: dict = {}, gestures=False):
        """Create epochs from raw data.

        :param raw: Raw data
        :type raw: RawArray
        :param epochs_params: Parameters for epoch creation
        :type epochs_params: dict
        :param gestures: Whether the data contains gestures or digits' data.
            False: digits' data, True: gestures' data, defaults to False
        :type gestures: bool, optional
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
                baseline=(None, 1.0),
                preload=True,
            )

        if gestures:
            return mne.Epochs(
                raw,
                self.events,
                self.event_id,
                tmin=-0.5,
                tmax=2.1,
                baseline=(None, 0.0),
                preload=True,
            )

        # Setting default parameters
        default_params = {}
        cue_ids = np.arange(1, 6)
        default_params["preload"] = True
        default_params["event_id"] = [2 ** (i + 2) for i in cue_ids]
        default_params["tmin"], default_params["tmax"] = (-2.0, 2.0)
        default_params.update(epochs_params)

        time_calculation = "avg"
        if time_calculation == "avg":
            tmin = default_params["tmin"]
            default_params["baseline"] = (
                tmin,
                self.calculate_avg_times(cue_ids=cue_ids),
            )

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

    def select_channels(self, raw: RawArray, channel_params: dict = None) -> np.ndarray:
        """Select the channels to use for the analysis.

        :param raw: Raw data
        :type raw: RawArray
        :param channel_params: Parameters for selecting channels
        :type channel_params: dict
        :return: name of the channels to use
        :rtype: np.ndarray
        """
        default_params = {
            "zscore_threshold_low": -1.5,
            "zscore_threshold_high": 1.5,
        }
        default_params.update(channel_params)
        bad_x = detect_bad_channels_by_zscore(
            raw,
            coordinate="X",
            zscore_low=default_params["zscore_threshold_low"],
            zscore_high=default_params["zscore_threshold_high"],
        )
        bad_y = detect_bad_channels_by_zscore(
            raw,
            coordinate="Y",
            zscore_low=default_params["zscore_threshold_low"],
            zscore_high=default_params["zscore_threshold_high"],
        )
        bad_z = detect_bad_channels_by_zscore(
            raw,
            coordinate="Z",
            zscore_low=default_params["zscore_threshold_low"],
            zscore_high=default_params["zscore_threshold_high"],
        )

        # Find the channels that are meg and not bad
        good_ch_indices = mne.pick_types(raw.info, meg=True, exclude="bads")
        good_ch_names = [raw.info["ch_names"][i] for i in good_ch_indices]

        return np.array(
            [
                name
                for name in good_ch_names
                if name not in bad_x and name not in bad_y and name not in bad_z
            ]
        )

    def manual_artifact_removal(
        self,
        raw: RawArray,
        channel_names: np.ndarray,
        events: np.ndarray,
        event_id: dict,
        epochs_params: dict,
        gestures: bool = False,
    ) -> Tuple[RawArray, mne.Epochs]:
        """Remove artifacts from the data.

        :param raw: Raw data
        :type raw: RawArray
        :param channel_names: Names of the channels to use
        :type channel_names: np.ndarray
        :param events: Events array
        :type events: np.ndarray
        :param event_id: Dictionary with event codes as keys and event names as values
        :type event_id: dict
        :param epochs_params: Parameters for epoch creation
        :type epochs_params: dict
        :param gestures: Whether the data contains gestures or digits' data.
            False: digits' data, True: gestures' data, defaults to False
        :type gestures: bool, optional
        :return: Raw data and epochs after artifact removal
        :rtype: Tuple[RawArray, mne.Epochs]
        """
        # Get the channel indices for the specified channels
        specified_channel_indices = [raw.ch_names.index(ch) for ch in channel_names]

        # Get the channel indices for channels not in channel_names
        remaining_channel_indices = [
            i for i, ch in enumerate(raw.ch_names) if ch not in channel_names
        ]

        # Mark the remaining channels as bad
        raw.info["bads"] = [raw.ch_names[i] for i in remaining_channel_indices]

        # Create the custom order
        custom_order = specified_channel_indices + remaining_channel_indices

        # Plot with the custom order
        raw.plot(block=True, events=events, event_id=event_id, order=custom_order)

        epochs_corrected = self.create_epochs(raw, epochs_params, gestures=gestures)

        return raw, epochs_corrected

    def calculate_avg_times(self, cue_ids: np.ndarray) -> float:
        """Calculate the average time between the cue and press events.

        :param cue_ids: Cue ids to consider
        :type cue_ids: np.ndarray
        :return: Average time between cue and press events
        :rtype: float
        """
        samp_freq = self.samp_freq
        events = self.events

        time_differences = []

        for cue_id in cue_ids:
            press_id = 2 ** (cue_id + 2)
            cue_indices = np.where(events[:, 2] == cue_id)[0]
            for cue_idx in cue_indices:
                # Find the next event that matches the press_id
                subsequent_events = events[cue_idx + 1 :, 2]
                try:
                    press_idx_rel = np.where(subsequent_events == press_id)[0][0]
                    press_idx = cue_idx + 1 + press_idx_rel
                    time_diff = (events[press_idx, 0] - events[cue_idx, 0]) / samp_freq
                    time_differences.append(time_diff)
                except IndexError:
                    # No corresponding press event found
                    pass

        return -np.mean(time_differences)
