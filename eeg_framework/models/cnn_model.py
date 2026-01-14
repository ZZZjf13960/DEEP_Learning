import torch
import torch.nn as nn
from .base_model import BaseModel

class SimpleEEGNet(BaseModel):
    """
    A simplified Convolutional Neural Network for EEG decoding.
    Inspired by EEGNet (Lawhern et al., 2018).
    """
    def __init__(self, n_channels, n_timepoints, n_classes, dropout=0.5):
        """
        Args:
            n_channels (int): Number of EEG channels.
            n_timepoints (int): Number of timepoints per sample.
            n_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(SimpleEEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # Layer 1: Temporal Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        # Layer 2: Spatial Convolution (Depthwise Conv)
        # Groups=8 means each of the 8 input channels is convolved with its own set of filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # Layer 3: Separable Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate the size of the flattened feature map
        # Input: (1, n_channels, n_timepoints)
        # Conv1: (8, n_channels, n_timepoints) (approx padding)
        # Conv2: (16, 1, n_timepoints) -> Pool(4) -> (16, 1, n_timepoints // 4)
        # Conv3: (16, 1, n_timepoints // 4) -> Pool(8) -> (16, 1, n_timepoints // 32)

        # Let's do a dummy forward pass to determine the input size of the fully connected layer
        dummy_input = torch.zeros(1, 1, n_channels, n_timepoints)
        with torch.no_grad():
            out = self.conv1(dummy_input)
            out = self.conv2(out)
            out = self.conv3(out)
            self.flatten_size = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_timepoints).
        """
        # Add channel dimension: (batch_size, 1, n_channels, n_timepoints)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
