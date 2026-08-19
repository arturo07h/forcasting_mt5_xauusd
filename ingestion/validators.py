"""Sanity checks on parsed M5 data — run after mql5_export_parser, before feature engineering."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    n_bars: int
    n_duplicate_timestamps: int
    n_ohlc_inconsistent: int
    n_zero_or_negative_spread: int
    n_bars_without_ticks: int
    gaps: list = field(default_factory=list)  # list of (start, end, missing_bars) beyond expected market closures


def validate(df: pd.DataFrame, expected_bar_seconds: int = 300) -> ValidationReport:
    n_duplicate_timestamps = int(df["time"].duplicated().sum())

    ohlc_bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    n_ohlc_inconsistent = int(ohlc_bad.sum())

    n_zero_or_negative_spread = int((df["spread_mean_points"] <= 0).sum())
    n_bars_without_ticks = int((df["tick_count"] == 0).sum())

    deltas = df["time"].diff().dt.total_seconds()
    gap_mask = deltas > expected_bar_seconds
    gaps = [
        (df["time"].iloc[i - 1], df["time"].iloc[i], int(deltas.iloc[i] // expected_bar_seconds) - 1)
        for i in deltas[gap_mask].index
    ]

    return ValidationReport(
        n_bars=len(df),
        n_duplicate_timestamps=n_duplicate_timestamps,
        n_ohlc_inconsistent=n_ohlc_inconsistent,
        n_zero_or_negative_spread=n_zero_or_negative_spread,
        n_bars_without_ticks=n_bars_without_ticks,
        gaps=gaps,
    )
