"""Monotonic multi-quantile output head — guarantees q05<=q10<=q50<=q90<=q95 by
construction (cumulative softplus offsets from the median), instead of relying on
post-hoc sorting to fix quantile crossing. Assumes an odd-length, sorted, symmetric-
around-the-median quantile list (true for [0.05, 0.10, 0.50, 0.90, 0.95]).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicQuantileHead(nn.Module):
    def __init__(self, in_features: int, quantiles: list[float]):
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.mid_idx = len(self.quantiles) // 2
        self.linear = nn.Linear(in_features, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.linear(x)  # (batch, n_quantiles)
        mid = raw[:, self.mid_idx]

        cols = [None] * len(self.quantiles)
        cols[self.mid_idx] = mid

        cur = mid
        for i in range(self.mid_idx + 1, len(self.quantiles)):
            cur = cur + F.softplus(raw[:, i])
            cols[i] = cur

        cur = mid
        for i in range(self.mid_idx - 1, -1, -1):
            cur = cur - F.softplus(raw[:, i])
            cols[i] = cur

        return torch.stack(cols, dim=1)  # (batch, n_quantiles), columns match sorted(quantiles)


def pinball_loss_multi(preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """preds: (batch, n_quantiles) matching sorted(quantiles); target: (batch,)."""
    target = target.unsqueeze(1)
    diff = target - preds
    q = torch.tensor(sorted(quantiles), device=preds.device, dtype=preds.dtype).unsqueeze(0)
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()
