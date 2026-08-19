"""On-the-fly windowed Dataset — never materializes all (N, seq_len, F) windows at once,
which would blow past 8GB RAM for 1M+ rows (e.g. 1.3M x 64 x 34 x 4 bytes ~ 11GB for one
fold alone). Each __getitem__ just slices the already-in-memory flat array.

Windows are row-based, not strictly time-based: since ~2-5% of rows were dropped
(feature warmup, gap exclusions), "the last 64 rows" occasionally spans slightly more
than 64*5=320 wall-clock minutes. Flagged here rather than engineered away — a
time-based windowing scheme would meaningfully complicate this for a small fraction of
sequences, not worth it for a first DL pass.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, i):
        window = self.X[i:i + self.seq_len]
        target = self.y[i + self.seq_len - 1]
        return window, target


def standardize(train_X: np.ndarray, *other_X: np.ndarray):
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = [(train_X - mean) / std]
    for X in other_X:
        out.append((X - mean) / std)
    return out if len(out) > 1 else out[0]
