
# ================================
# 5. TASK-SPECIFIC CLASSIFICATION BRANCHES
# ================================
import torch
import torch.nn as nn


class ClassificationBranch(nn.Module):
    """
    Task-specific classification branch
    """

    def __init__(self, input_dim, hidden_dim=128, lstm_units=128):
        super(ClassificationBranch, self).__init__()

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_units * 2, 64),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, time, features)

        # Feature projection
        x = self.feature_projection(x)  # (batch, time, hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, lstm_units*2)

        # Global average pooling
        x = torch.mean(lstm_out, dim=1)  # (batch, lstm_units*2)

        # Final classification
        output = self.fc_layers(x)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)
