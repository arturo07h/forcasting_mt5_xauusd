"""H1 counterpart of walk_forward.py. Same calendar fold boundaries (regime coverage
reasoning unchanged), purge/embargo recomputed in H1-bar terms: label horizon = 6 bars,
embargo = 48 bars (~2 days) — a conservative round number given |return| autocorrelation
at H1 was still ~0.09 at lag 14 and hadn't clearly decayed below 0.05 within the tested
range, unlike M5's cleaner decay point.
"""
import pandas as pd

LABEL_HORIZON_BARS = 6
EMBARGO_BARS = 48
PURGE_BARS = LABEL_HORIZON_BARS + EMBARGO_BARS  # 54 H1 bars ≈ 2.25 days

FOLDS = [
    {"train_end": "2014-01-01", "test_start": "2014-01-01", "test_end": "2016-01-01"},
    {"train_end": "2016-01-01", "test_start": "2016-01-01", "test_end": "2018-01-01"},
    {"train_end": "2018-01-01", "test_start": "2018-01-01", "test_end": "2020-01-01"},
    {"train_end": "2020-01-01", "test_start": "2020-01-01", "test_end": "2022-01-01"},
    {"train_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2024-01-01"},
    {"train_end": "2024-01-01", "test_start": "2024-01-01", "test_end": "2026-08-19"},
]


def fold_masks(time: pd.Series, fold: dict):
    train_end = pd.Timestamp(fold["train_end"], tz="UTC")
    test_start = pd.Timestamp(fold["test_start"], tz="UTC")
    test_end = pd.Timestamp(fold["test_end"], tz="UTC")

    purge_cutoff = train_end - pd.Timedelta(hours=PURGE_BARS)
    train_mask = time < purge_cutoff
    test_mask = (time >= test_start) & (time < test_end)
    return train_mask, test_mask


def describe_folds(time: pd.Series) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(time, fold)
        rows.append({
            "fold": i, "train_end": fold["train_end"], "test_start": fold["test_start"], "test_end": fold["test_end"],
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        })
    return pd.DataFrame(rows)
