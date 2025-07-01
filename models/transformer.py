
# ================================
# 4. TRANSFORMER ENCODER
# ================================
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal modeling
    """

    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size, channels, time_steps = x.size()

        # Reshape to (batch, time, channels) for transformer
        x = x.permute(0, 2, 1)  # (batch, time, channels)

        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch, time, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        x = self.transformer(x)  # (batch, time, d_model)

        return x
