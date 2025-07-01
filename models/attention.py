# ================================
# 2. ATTENTION MECHANISMS
# ================================
import torch
import torch.nn as nn


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size, channels, _ = x.size()

        # Squeeze: Global average pooling
        y = self.global_pool(x).view(batch_size, channels)

        # Excitation: Two fully connected layers with bottleneck
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Scale: Channel-wise multiplication
        y = y.view(batch_size, channels, 1)
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, time)

        # Compute spatial features
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Concatenate and apply convolution
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)

        return x * attention


class TaskSpecificAttention(nn.Module):
    """
    Task-specific attention combining SE and spatial attention
    """

    def __init__(self, channels):
        super(TaskSpecificAttention, self).__init__()
        self.se_block = SqueezeExcitationBlock(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.se_block(x)
        x = self.spatial_attention(x)
        return x
