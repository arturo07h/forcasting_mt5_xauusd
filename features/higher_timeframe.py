"""Merges higher-timeframe context (XAUUSD H1/H4 trend structure, DXY/USTBOND) onto
the M5 timeline — cross-asset and same-asset alike, same merge mechanics.

Leakage trap this exists to avoid: an H1 bar timestamped at its OPEN (Dukascopy
convention) isn't actually *closed* — its high/low/close aren't known yet — until an
hour later. A naive backward as-of join on raw timestamps would let an M5 bar at 14:35
see the still-forming 14:00 H1 bar's OHLC. Every higher-TF frame is shifted to its
close time before joining, so only fully-closed bars are ever visible.
"""
import numpy as np
import pandas as pd


def _available_at_close(df: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    out = df.copy()
    out["available_time"] = out["time"] + pd.Timedelta(seconds=bar_seconds)
    return out.sort_values("available_time")


def trend_features(htf_df: pd.DataFrame, bar_seconds: int, prefix: str,
                    windows=(4, 12)) -> pd.DataFrame:
    """log-return over each window (in bars) and a z-score of price vs its rolling mean,
    computed on the closed-bar series before the as-of join happens.
    """
    df = htf_df.sort_values("time").copy()
    close_col = "close" if "close" in df.columns else "bid_close"
    df["log_ret"] = np.log(df[close_col]).diff()

    feats = {"time": df["time"], close_col: df[close_col]}
    for w in windows:
        feats[f"{prefix}_ret_{w}"] = np.log(df[close_col] / df[close_col].shift(w))
        roll_mean = df[close_col].rolling(w).mean()
        roll_std = df[close_col].rolling(w).std()
        feats[f"{prefix}_zscore_{w}"] = (df[close_col] - roll_mean) / roll_std
    feat_df = pd.DataFrame(feats)
    return _available_at_close(feat_df, bar_seconds)


def merge_asof_closed(m5_time: pd.Series, htf_features: pd.DataFrame,
                       feature_cols: list[str]) -> pd.DataFrame:
    """m5_time must be sorted ascending. Returns feature_cols aligned to m5_time, using
    only htf bars that were fully closed by that timestamp.
    """
    left = pd.DataFrame({"time": m5_time}).sort_values("time")
    left["time"] = left["time"].astype("datetime64[us, UTC]")
    right = htf_features[["available_time"] + feature_cols].copy()
    right["available_time"] = right["available_time"].astype("datetime64[us, UTC]")
    merged = pd.merge_asof(
        left, right,
        left_on="time", right_on="available_time", direction="backward",
    )
    return merged[feature_cols].set_index(left.index)
