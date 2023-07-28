import numpy as np
from typing import Tuple


class Preprocessing:
    def __init__(
        self,
        raw,
        events: np.ndarray,
        events_id: dict,
        signal_sep_params: dict = None,
        artifact_params: dict = None,
        segmentation_params: dict = None,
    ) -> None:
        self.raw = raw

        self.events, self.events_id, self.wrong_trials = self.preprocess_events(
            events, events_id
        )
        artifact_trials, self.raw = self.remove_artifacts()
        self.wrong_trials.extend(artifact_trials)

        # self.data = self.signal_space_separation(signal_sep_params)
        # self.data = self.normalize_data()
        # self.data = self.trial_segmentation()

    def preprocess_events(
        self, events: np.ndarray, events_id: dict
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
        :return: Preprocessed events array
        :rtype: np.ndarray
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
            wrong_previous_trial.append(trial_idx)

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

        events_id[6] = "cue_not_answered"
        events_id[9] = "cue_wrong_button"
        events_id[10] = "cue_multiple_buttons"
        events_id[11] = "press_wrong_button"
        events_id[12] = "press_multiple_buttons"

        return events, events_id, wrong_previous_trial

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
        data = self.raw

        return artifact_trials, data

    def normalize_data(self):
        pass

    def trial_segmentation(self, segmentation_params: dict = None):
        pass
