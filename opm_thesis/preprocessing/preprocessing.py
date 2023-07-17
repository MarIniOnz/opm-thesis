class Preprocessing:
    def __init__(
        self,
        raw,
        signal_sep_params: dict,
        artifact_params: dict,
        segmentation_params: dict,
    ) -> None:
        self.raw = raw

        self.data = self.signal_space_separation(signal_sep_params)
        self.artifact_trials, self.data = self.remove_artifacts()
        self.data = self.normalize_data()
        self.data = self.trial_segmentation()

        pass

    def signal_space_separation(self):
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

    def trial_segmentation(self):
        pass
