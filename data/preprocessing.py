import numpy as np


def preprocess_fnirs_data(raw_data, fs=10.0):
    """
    Preprocess fNIRS data with standard techniques

    Args:
        raw_data: Raw fNIRS signals (channels × time points)
        fs: Sampling frequency in Hz

    Returns:
        Preprocessed fNIRS data
    """
    # 1. Motion artifact removal using Savitzky-Golay filter
    from scipy.signal import savgol_filter
    data_filtered = savgol_filter(
        raw_data, window_length=int(fs*3), polyorder=3, axis=1)

    # 2. Bandpass filtering (0.01-0.2 Hz) to remove physiological noise
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low, high = 0.01 / nyq, 0.2 / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    data_filtered = filtfilt(b, a, data_filtered, axis=1)

    # 3. Signal detrending to remove slow drifts
    from scipy import signal
    data_filtered = signal.detrend(data_filtered, axis=1)

    # 4. Z-score normalization for each channel
    data_normalized = np.zeros_like(data_filtered)
    for i in range(data_filtered.shape[0]):
        channel_data = data_filtered[i, :]
        data_normalized[i, :] = (
            channel_data - np.mean(channel_data)) / np.std(channel_data)

    return data_normalized
