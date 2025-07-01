# ================================
# 6. MAIN DUALNET ARCHITECTURE
# ================================
import torch.nn as nn

from models.attention import TaskSpecificAttention
from models.classifier import ClassificationBranch
from models.convnet import MultiScaleConvNet
from models.transformer import TransformerEncoder


class DualNet(nn.Module):
    """
    Unified architecture for simultaneous pain and stress detection
    """

    def __init__(self, input_channels, d_model=256, nhead=8, num_layers=2):
        super(DualNet, self).__init__()

        # Shared feature extraction
        self.shared_extractor = MultiScaleConvNet(input_channels, filters=64)
        shared_channels = self.shared_extractor.output_channels  # 192

        # Task-specific attention mechanisms
        self.pain_attention = TaskSpecificAttention(shared_channels)
        self.stress_attention = TaskSpecificAttention(shared_channels)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            input_dim=shared_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

        # Task-specific classification branches
        self.pain_classifier = ClassificationBranch(
            d_model, hidden_dim=128, lstm_units=128)
        self.stress_classifier = ClassificationBranch(
            d_model, hidden_dim=128, lstm_units=128)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, task=None):
        """
        Forward pass through DualNet

        Args:
            x: Input fNIRS signals (batch, channels, time)
            task: Task type ('pain', 'stress', or None for both)

        Returns:
            Dictionary with predictions for requested tasks
        """
        # Shared feature extraction
        shared_features = self.shared_extractor(x)  # (batch, 192, time/2)

        # Apply task-specific attention
        pain_features = self.pain_attention(shared_features)
        stress_features = self.stress_attention(shared_features)

        results = {}

        if task is None or task == 'pain':
            # Pain pathway
            pain_encoded = self.transformer(
                pain_features)  # (batch, time, d_model)
            pain_pred = self.pain_classifier(pain_encoded)
            results['pain'] = pain_pred

        if task is None or task == 'stress':
            # Stress pathway
            stress_encoded = self.transformer(
                stress_features)  # (batch, time, d_model)
            stress_pred = self.stress_classifier(stress_encoded)
            results['stress'] = stress_pred

        return results
