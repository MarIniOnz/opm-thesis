"""Utils for preprocessing."""
from typing import Tuple, List

import mne
import numpy as np
from mne import Info
from mne.io import RawArray


def get_closest_sensors(
    info: Info, centre_channel: str, num_channels: int, include_xyz: bool = True
) -> np.ndarray:
    """
    Get the closest sensors to a specified center sensor.

    This function retrieves the closest sensors based on Euclidean distance in
    2D space (ignoring the z dimension in sensor locations). It optionally
    includes the corresponding `[Y]` and `[Z]` sensors if they exist.

    :param info: The info object of data.
    :type info: mne.Info
    :param centre_channel: The name of the centre channel to which distances will be
        calculated.
    :type centre_channel: str
    :param num_channels: The number of closest channels to retrieve.
    :type num_channels: int
    :param include_xyz: Whether to include corresponding `[Y]` and `[Z]` channels, by
        default True.
    :type include_xyz: bool, optional

    :return: An array of channel names representing the closest channels to the
             specified centre channel.
    :rtype: np.ndarray

    Example
    -------
    >>> closest_sensors = get_closest_sensors(raw, 'LQ[X]', 25)
    """
    coordinate = centre_channel[-2]

    # Extract sensor positions and names
    sensor_positions = mne.find_layout(info).pos
    sensor_names_with_pos = np.array(mne.find_layout(info).names)

    # Filter sensors that contain 'coordinate in their name
    mask = np.array(["[" + coordinate + "]" in name for name in sensor_names_with_pos])
    sensor_positions_coordinate = sensor_positions[mask]
    sensor_names_with_pos_coordinate = sensor_names_with_pos[mask]

    # Find index of Centre channel
    centre_idx = np.where(sensor_names_with_pos_coordinate == centre_channel)[0][0]

    # Calculate Euclidean distance from the centre channel to all sensors
    distances = np.linalg.norm(
        sensor_positions_coordinate - sensor_positions_coordinate[centre_idx], axis=1
    )

    # Find the closest sensors
    closest_sensors_idx = np.argsort(distances)[:num_channels]

    # Get the names of the closest sensors
    closest_sensors_names = sensor_names_with_pos_coordinate[closest_sensors_idx]

    # Optionally, add the corresponding rest of the channels
    rest_coordinates = list(set(["X", "Y", "Z"]) - set([coordinate]))
    if include_xyz:
        names_2 = [
            name[:-3] + "[" + rest_coordinates[0] + "]"
            for name in closest_sensors_names
        ]
        names_3 = [
            name[:-3] + "[" + rest_coordinates[1] + "]"
            for name in closest_sensors_names
        ]
        return np.concatenate((closest_sensors_names, names_2, names_3))

    else:
        return closest_sensors_names


def detect_bad_channels_by_zscore(
    raw: RawArray, coordinate: str, zscore_threshold: float = 2.0
) -> List[str]:
    """
    Detect bad channels based on z-score of the channel's data.

    :param raw: mne.io.RawArray
    :type raw: mne.io.RawArray
    :param coordinate: Coordinate to use for detecting bad channels. Should be one of
        ['X', 'Y', 'Z'].
    :type coordinate: str
    :param zscore_threshold: Threshold for z-score to be marked as bad. Default is 2.0.
    :type zscore_threshold: float

    :return: List of channel names marked as bad.
    :rtype: List[str]
    """
    if coordinate not in ["X", "Y", "Z"]:
        raise ValueError("Invalid coordinate. Should be one of ['X', 'Y', 'Z']")

    # Filter channels by coordinate
    ch_names = np.array(raw.ch_names)
    mask = np.array([f"[{coordinate}]" in ch_name for ch_name in ch_names])
    selected_ch_names = ch_names[mask]
    selected_data = raw._data[mask, :]

    # Compute peak-to-peak amplitude for selected channels
    ptp_values = np.ptp(selected_data, axis=1)

    # Compute z-scores for each channel
    z_scores = (ptp_values - np.mean(ptp_values)) / np.std(ptp_values)

    # Detect channels with z-scores greater than the threshold
    bad_channels = np.where(np.abs(z_scores) > zscore_threshold)[0]
    bad_channel_names = selected_ch_names[bad_channels].tolist()

    return bad_channel_names


def calculate_max_times(
    events: np.ndarray,
    sample_frequency: float,
    cue_ids: np.ndarray,
    mult_factor: float = 1.1,
) -> Tuple[float, float]:
    """Calculate the minimum and maximum time before and after the cue events.

    :param events: Events array
    :type events: np.ndarray
    :param sample_frequency: Sampling frequency of the data
    :type sample_frequency: float
    :param cue_ids: Cue ids to consider
    :type cue_ids: np.ndarray
    :param mult_factor: Multiplication factor for the minimum time before the cue
        event, defaults to 1.1
    :type mult_factor: float, optional
    :return: Minimum and maximum time before and after the cue events
    :rtype: Tuple[float, float]
    """

    # All the indices where any of the cue_ids is present
    indices = np.where(np.isin(events[:, 2], cue_ids))[0]

    t_max = 1 / sample_frequency
    t_min = events[-1, 0] / sample_frequency
    for i in indices:
        time_after = (events[i + 1, 0] - events[i, 0]) / sample_frequency
        if time_after > t_max:
            t_max = time_after

        time_before = (events[i, 0] - events[i - 1, 0]) / sample_frequency
        if time_before < t_min:
            t_min = time_before
    t_min, t_max = t_min * mult_factor, t_max * mult_factor
    print(
        "The times for epochs segmentation are \n"
        f"t_min: {t_min}, t_max: {t_max} when using MaxTimes"
    )
    return -t_min, t_max


def calculate_avg_times(
    events: np.ndarray,
    sample_frequency: float,
    cue_ids: np.ndarray,
    mult_factor: float = 1.1,
) -> Tuple[float, float]:
    """Calculate the minimum and maximum time before and after the cue events.

    :param events: Events array
    :type events: np.ndarray
    :param sample_frequency: Sampling frequency of the data
    :type sample_frequency: float
    :param cue_ids: Cue ids to consider
    :type cue_ids: np.ndarray
    :param mult_factor: Multiplication factor for the minimum time before the cue
        event, defaults to 1.1
    :type mult_factor: float, optional
    :return: Minimum and maximum time before and after the cue events
    :rtype: Tuple[float, float]
    """
    indices = np.where(np.isin(events[:, 2], cue_ids))[0]
    time_after = []
    time_before = []
    for i in indices:
        time_after.append((events[i + 1, 0] - events[i, 0]) / sample_frequency)
        time_before.append((events[i, 0] - events[i - 1, 0]) / sample_frequency)

    t_min, t_max = (
        np.mean(time_before),
        np.mean(time_after) * mult_factor + np.std(time_after),
    )
    print(
        "The times for epochs segmentation are \n"
        f"t_min: {t_min}, t_max: {t_max} when using AvgTimes"
    )

    return -t_min, t_max


def create_fixed_length_events(raw, duration):
    """
    Create a set of events that splits the raw data into non-overlapping epochs.

    Parameters:
    - raw: mne.io.Raw
        The continuous data.
    - duration: float
        Duration of each epoch in seconds.

    Returns:
    - events: np.ndarray
        The artificial events array.
    """
    event_id = {"Fixed_length": 999}
    start_samp = raw.first_samp
    stop_samp = raw.last_samp

    # Generate event time points (in samples) by stepping through the recording
    # from start to stop with steps of size (duration in samples)
    duration_samp = int(duration * raw.info["sfreq"])
    event_times = np.arange(start_samp, stop_samp, duration_samp)

    # Create an events array
    events = np.vstack(
        (
            event_times,
            np.zeros(len(event_times), dtype=int),
            event_id["Fixed_length"] * np.ones(len(event_times), dtype=int),
        )
    ).T

    return events, event_id


def create_resting_epochs(raw, duration):
    """
    Create epochs of fixed length.

    Parameters:
    - raw: mne.io.Raw
        The continuous data.
    - duration: float
        Duration of each epoch in seconds.

    Returns:
    - epochs: mne.Epochs
        The epochs data.
    """
    events, event_id = create_fixed_length_events(raw, duration)

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=0,
        tmax=duration,
        baseline=None,
        detrend=1,
        preload=True,
    )

    return epochs
