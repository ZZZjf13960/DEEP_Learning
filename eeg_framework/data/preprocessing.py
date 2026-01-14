import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EEGPreprocessor:
    """
    Class for EEG data preprocessing.
    """
    def __init__(self, sfreq=250):
        """
        Args:
            sfreq (int): Sampling frequency in Hz.
        """
        self.sfreq = sfreq

    def bandpass_filter(self, data, lowcut, highcut, order=5):
        """
        Apply a bandpass filter to the data.

        Args:
            data (np.ndarray): Data of shape (n_samples, n_channels, n_timepoints).
            lowcut (float): Low cutoff frequency.
            highcut (float): High cutoff frequency.
            order (int): Order of the filter.

        Returns:
            np.ndarray: Filtered data.
        """
        print(f"Applying bandpass filter: {lowcut}-{highcut} Hz")
        nyq = 0.5 * self.sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        # Apply filter along the time axis (last axis)
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

    def standardize(self, data):
        """
        Standardize data (z-score normalization) per channel.

        Args:
            data (np.ndarray): Data of shape (n_samples, n_channels, n_timepoints).

        Returns:
            np.ndarray: Standardized data.
        """
        print("Standardizing data...")
        n_samples, n_channels, n_timepoints = data.shape
        # Reshape to (n_samples * n_timepoints, n_channels) for StandardScaler which expects (n_samples, n_features)
        # But usually we standardize per channel across time, or per sample per channel.
        # Let's standardize per channel across all samples/timepoints to keep channel statistics consistent.

        # Transpose to (n_samples, n_timepoints, n_channels)
        data_transposed = data.transpose(0, 2, 1)
        # Flatten: (n_samples * n_timepoints, n_channels)
        data_reshaped = data_transposed.reshape(-1, n_channels)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)

        # Reshape back
        data_scaled = data_scaled.reshape(n_samples, n_timepoints, n_channels)
        # Transpose back to (n_samples, n_channels, n_timepoints)
        return data_scaled.transpose(0, 2, 1)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Args:
            X (np.ndarray): Data.
            y (np.ndarray): Labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed.

        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"Splitting data: test_size={test_size}")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
