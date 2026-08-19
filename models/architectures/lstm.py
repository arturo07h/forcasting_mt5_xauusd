"""LSTM baseline — simpler, well-understood, the DL comparison point for the TCN."""
import torch
import torch.nn as nn

from models.architectures.quantile_heads import MonotonicQuantileHead


class LSTMQuantile(nn.Module):
    def __init__(self, n_features: int, quantiles: list[float], hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = MonotonicQuantileHead(hidden_size, quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
