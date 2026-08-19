"""Expanding-window walk-forward folds with purge + embargo (López de Prado) — never
standard k-fold on this data. 6 folds, ~2-year test windows, chosen to spread real
regime coverage across folds: fold 1's train set already includes the 2008 GFC, fold 4's
test set covers the 2020 COVID crash, fold 6's test set covers the 2024-2026 rally.

Purge width = label horizon (12 candles) + embargo (50 candles, ~250 min — the EDA's
|return| autocorrelation decay point) = 62 candles removed from the end of each fold's
training set, so no training label window overlaps the test period.
"""
import pandas as pd

LABEL_HORIZON_BARS = 12
EMBARGO_BARS = 50
PURGE_BARS = LABEL_HORIZON_BARS + EMBARGO_BARS  # 62 bars = ~5h10m

FOLDS = [
    {"train_end": "2014-01-01", "test_start": "2014-01-01", "test_end": "2016-01-01"},
    {"train_end": "2016-01-01", "test_start": "2016-01-01", "test_end": "2018-01-01"},
    {"train_end": "2018-01-01", "test_start": "2018-01-01", "test_end": "2020-01-01"},
    {"train_end": "2020-01-01", "test_start": "2020-01-01", "test_end": "2022-01-01"},  # COVID in test
    {"train_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2024-01-01"},
    {"train_end": "2024-01-01", "test_start": "2024-01-01", "test_end": "2026-08-19"},  # 2024-26 rally in test
]


def fold_masks(time: pd.Series, fold: dict):
    train_end = pd.Timestamp(fold["train_end"], tz="UTC")
    test_start = pd.Timestamp(fold["test_start"], tz="UTC")
    test_end = pd.Timestamp(fold["test_end"], tz="UTC")

    purge_cutoff = train_end - pd.Timedelta(minutes=5 * PURGE_BARS)
    train_mask = time < purge_cutoff
    test_mask = (time >= test_start) & (time < test_end)
    return train_mask, test_mask


def describe_folds(time: pd.Series) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(time, fold)
        rows.append({
            "fold": i,
            "train_end": fold["train_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
    return pd.DataFrame(rows)
