import mne
import numpy as np
from typing import Tuple


class Preprocessing:
    def __init__(
        self,
        raw,
        events: np.ndarray,
        event_id: dict,
        filter_params: dict = {},
        epochs_params: dict = {},
        signal_sep_params: dict = {},
        artifact_params: dict = {},
        segmentation_params: dict = {},
    ) -> None:
        self.raw = raw
        assert "sfreq" in raw.info, "Sampling frequency not found in raw.info"
        self.samp_freq = raw.info["sfreq"]

        self.raw = self.apply_filters(filter_params)
        self.events, self.event_id, self.wrong_trials = self.preprocess_events(
            events, event_id
        )
        self.epochs = self.create_epochs(epochs_params)
        artifact_trials = self.remove_artifacts()
        self.wrong_trials.extend(artifact_trials)

        # self.data = self.signal_space_separation(signal_sep_params)
        # self.data = self.normalize_data()

    def apply_filters(self, filter_params: dict):
        """Apply filters to raw data.

        :param filter_params: Parameters for filtering
        :type filter_params: dict
        :return: Filtered raw data
        :rtype: mne.io.Raw
        """
        default_params = dict(
            {"l_freq": 0.01, "h_freq": 330, "fir_design": "firwin", "phase": "zero"}
        )
        default_params.update(filter_params)

        notch_freq = 50
        raw = self.raw.notch_filter(freqs=notch_freq, fir_design="firwin")

        return raw.filter(**default_params)

    def preprocess_events(
        self, events: np.ndarray, event_id: dict
    ) -> Tuple[np.ndarray, dict]:
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

    def create_epochs(self, epochs_params: dict):
        """Create epochs from raw data.

        :param epochs_params: Parameters for epoch creation
        :type epochs_params: dict
        :return: Epochs object
        :rtype: mne.Epochs
        """
        # Setting default parameters
        default_params = dict()
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
            epochs_params["tmin"], epochs_params["tmax"] = self.calculate_max_times(
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
            self.raw,
            events=self.events,
            **default_params,
        )

    def signal_space_separation(self, signal_sep_params: dict = None):
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

        artifact_trials = []

        return artifact_trials

    def normalize_data(self):
        pass

    def calculate_max_times(
        self, cue_ids: np.ndarray, mult_factor: float = 1.1
    ) -> Tuple[float, float]:
        """Calculate the minimum and maximum time before and after the cue events.

        :param cue_ids: Cue ids to consider
        :type cue_ids: np.ndarray
        :param mult_factor: Multiplication factor for the minimum time before the cue
            event, defaults to 1.1
        :type mult_factor: float, optional
        :return: Minimum and maximum time before and after the cue events
        :rtype: Tuple[float, float]
        """
        samp_freq = self.samp_freq
        events = self.events

        # All the indices where any of the cue_ids is present
        indices = np.where(np.isin(events[:, 2], cue_ids))[0]

        t_max = 1 / samp_freq
        t_min = self.events[-1, 0] / samp_freq
        for i in indices:
            time_after = (events[i + 1, 0] - events[i, 0]) / samp_freq
            if time_after > t_max:
                t_max = time_after

            time_before = (events[i, 0] - events[i - 1, 0]) / samp_freq
            if time_before < t_min:
                t_min = time_before
        t_min, t_max = t_min * mult_factor, t_max * mult_factor
        print(
            "The times for epochs segmentation are \n"
            f"t_min: {t_min}, t_max: {t_max} when using MaxTimes"
        )
        return -t_min, t_max

    def calculate_avg_times(
        self, cue_ids: np.ndarray, mult_factor: float = 1.1
    ) -> Tuple[float, float]:
        """Calculate the minimum and maximum time before and after the cue events using
        the average time before and after the cue events.

        :param cue_ids: Cue ids to consider
        :type cue_ids: np.ndarray
        :param mult_factor: Multiplication factor for the minimum time before the cue
            event, defaults to 1.1
        :type mult_factor: float, optional
        :return: Minimum and maximum time before and after the cue events
        :rtype: Tuple[float, float]
        """
        samp_freq = self.samp_freq
        events = self.events

        indices = np.where(np.isin(events[:, 2], cue_ids))[0]
        time_after = []
        time_before = []
        for i in indices:
            time_after.append((events[i + 1, 0] - events[i, 0]) / samp_freq)
            time_before.append((events[i, 0] - events[i - 1, 0]) / samp_freq)

        t_min, t_max = (
            np.mean(time_before),
            np.mean(time_after) * mult_factor + np.std(time_after),
        )
        print(
            "The times for epochs segmentation are \n"
            f"t_min: {t_min}, t_max: {t_max} when using AvgTimes"
        )

        return -t_min, t_max
