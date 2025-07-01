# ================================
# 3. SHARED FEATURE EXTRACTION MODULE
# ================================
import torch
import torch.nn as nn


class MultiScaleConvNet(nn.Module):
    """
    Multi-scale convolutional network for feature extraction
    """

    def __init__(self, input_channels, filters=64):
        super(MultiScaleConvNet, self).__init__()

        # Three parallel pathways with different kernel sizes
        self.pathway1 = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.pathway2 = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.pathway3 = nn.Sequential(
            nn.Conv1d(input_channels, filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.output_channels = filters * 3  # Concatenated channels

    def forward(self, x):
        # x shape: (batch, channels, time)
        p1 = self.pathway1(x)
        p2 = self.pathway2(x)
        p3 = self.pathway3(x)

        # Concatenate along channel dimension
        out = torch.cat([p1, p2, p3], dim=1)
        return out
