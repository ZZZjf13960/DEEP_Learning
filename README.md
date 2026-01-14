# Deep Learning for EEG Framework

A modular, GPU-accelerated framework for EEG signal decoding. This project implements a complete end-to-end pipeline, including data generation/loading, preprocessing, model building (CNN), and training/evaluation.

## Project Structure

```
eeg_framework/
├── data/
│   ├── loader.py          # Data generation and PyTorch DataLoaders
│   └── preprocessing.py   # Signal processing (Bandpass filter, Standardization)
├── models/
│   ├── base_model.py      # Abstract base class
│   └── cnn_model.py       # SimpleEEGNet implementation
├── training/
│   ├── trainer.py         # Training loop, validation, and GPU handling
│   └── metrics.py         # Evaluation metrics (Accuracy, F1, Confusion Matrix)
└── utils/                 # Utility functions
```

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements:**
- numpy
- torch
- scipy
- scikit-learn
- matplotlib

## Usage

The project is designed to be runnable out-of-the-box. The entry point is `main.py`, which orchestrates the entire pipeline.

### Running the Pipeline

To train the model on synthetic EEG data:

```bash
python main.py
```

### What Happens When You Run It?

1.  **Data Loading:** The script generates synthetic EEG data (or loads it if you extend `loader.py`).
2.  **Preprocessing:**
    *   Applies a Bandpass Filter (4-40 Hz).
    *   Standardizes the data (Z-score normalization).
    *   Splits into Train/Test sets.
3.  **Model Initialization:** Initializes `SimpleEEGNet` (a CNN optimized for EEG).
4.  **GPU Detection:** Automatically detects if a CUDA-enabled GPU is available.
    *   If yes: Moves model and data to GPU and enables `cudnn.benchmark` for speed.
    *   If no: Falls back to CPU.
5.  **Training:** Trains the model for 5 epochs and outputs training/validation loss and accuracy.

### extending the Framework

-   **Use Your Own Data:** Modify `eeg_framework/data/loader.py` to load your `.edf` or `.set` files.
-   **Change the Model:** Add new architectures in `eeg_framework/models/` inheriting from `BaseModel`.
-   **Adjust Preprocessing:** Update `eeg_framework/data/preprocessing.py` to add new filters or artifact removal techniques.
