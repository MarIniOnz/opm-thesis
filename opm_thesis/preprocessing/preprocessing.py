import numpy as np

class Preprocessing:
    def __init__(
        self,
        raw,
        events,
        events_id,
        signal_sep_params: dict,
        artifact_params: dict,
        segmentation_params: dict,
    ) -> None:
        self.raw = raw
        self.events = events

        self.events = self.preprocess_events(events)
        self.artifact_trials, self.data = self.remove_artifacts()
        # self.data = self.signal_space_separation(signal_sep_params)
        # self.data = self.normalize_data()
        # self.data = self.trial_segmentation()

    def preprocess_events(self, events: np.ndarray) -> np.ndarray:
        # Check if the triggers follow the correct pattern
        # for i in range(len(events)):
        pass


        # return events

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
        data = self.data

        return artifact_trials, data

    def normalize_data(self):
        pass

    def trial_segmentation(self, segmentation_params: dict = None):
        pass
