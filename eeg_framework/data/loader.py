import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    """
    A PyTorch Dataset for EEG data.
    """
    def __init__(self, data, labels):
        """
        Args:
            data (np.ndarray): EEG data of shape (n_samples, n_channels, n_timepoints).
            labels (np.ndarray): Labels of shape (n_samples,).
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DataImporter:
    """
    Class to handle data loading. Currently supports synthetic data generation.
    """
    def __init__(self):
        pass

    def load_synthetic_data(self, n_samples=200, n_channels=16, n_timepoints=512, n_classes=2):
        """
        Generates synthetic EEG data.

        Args:
            n_samples (int): Number of samples.
            n_channels (int): Number of EEG channels.
            n_timepoints (int): Number of timepoints per sample.
            n_classes (int): Number of classes.

        Returns:
            X (np.ndarray): Synthetic EEG data (n_samples, n_channels, n_timepoints).
            y (np.ndarray): Synthetic labels (n_samples,).
        """
        print(f"Generating synthetic data: {n_samples} samples, {n_channels} channels, {n_timepoints} timepoints.")
        X = np.random.randn(n_samples, n_channels, n_timepoints)
        y = np.random.randint(0, n_classes, size=n_samples)

        # Add a simple pattern to make the data separable
        # For class 1, add a sine wave to the first few channels
        t = np.linspace(0, 1, n_timepoints)
        signal_freq = 10.0 # 10 Hz alpha wave

        for i in range(n_samples):
            if y[i] == 1:
                # Add signal to first half of channels
                for ch in range(n_channels // 2):
                    X[i, ch, :] += 2.0 * np.sin(2 * np.pi * signal_freq * t)

        return X, y

    def get_dataloader(self, X, y, batch_size=32, shuffle=True):
        """
        Creates a PyTorch DataLoader.

        Args:
            X (np.ndarray): Data.
            y (np.ndarray): Labels.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader.
        """
        dataset = EEGDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
