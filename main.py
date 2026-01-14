import torch
import torch.nn as nn
import torch.optim as optim
from eeg_framework.data.loader import DataImporter
from eeg_framework.data.preprocessing import EEGPreprocessor
from eeg_framework.models.cnn_model import SimpleEEGNet
from eeg_framework.training.trainer import Trainer

def main():
    # 1. Load Data
    importer = DataImporter()
    # Generate synthetic data
    X, y = importer.load_synthetic_data(n_samples=200, n_channels=16, n_timepoints=512, n_classes=2)

    # 2. Preprocess Data
    preprocessor = EEGPreprocessor(sfreq=250)
    # Bandpass filter 4-40Hz
    X = preprocessor.bandpass_filter(X, 4, 40)
    # Standardize
    X = preprocessor.standardize(X)

    # Split
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Create DataLoaders
    train_loader = importer.get_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = importer.get_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    # 3. Initialize Model
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {device}")
    else:
        device = 'cpu'
        print("GPU not detected. Falling back to CPU.")
        print(f"Using device: {device}")

    n_channels = X_train.shape[1]
    n_timepoints = X_train.shape[2]
    n_classes = 2

    model = SimpleEEGNet(n_channels, n_timepoints, n_classes)

    # 4. Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device=device)
    trainer.train(num_epochs=5)

if __name__ == "__main__":
    main()
