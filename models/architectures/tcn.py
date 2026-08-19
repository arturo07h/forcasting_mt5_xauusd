"""Temporal Convolutional Network — the primary DL candidate per the Phase 2 plan
(dilated causal convolutions, easier to keep strictly causal than an RNN, generally
better-behaved training than LSTM at this data scale)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architectures.quantile_heads import MonotonicQuantileHead


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad, 0))  # left-pad only — causal, no future leakage
        out = self.conv(padded)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TCN(nn.Module):
    def __init__(self, n_features: int, quantiles: list[float], channels: int = 32,
                 n_layers: int = 5, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_layers):
            layers.append(CausalConvBlock(in_ch, channels, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.net = nn.Sequential(*layers)
        self.head = MonotonicQuantileHead(channels, quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        h = self.net(x.transpose(1, 2))
        last = h[:, :, -1]
        return self.head(last)
