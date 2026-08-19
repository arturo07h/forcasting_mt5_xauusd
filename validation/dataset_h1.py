"""H1 counterpart of dataset.py. Target: forward N=6-candle (~6h) log-return, gap-aware
same as the M5 version (a window is only labeled if it doesn't cross a data gap).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR

N_HORIZON = 6

CORE_FEATURE_COLS = [
    "price_ffd_zscore", "rv_6", "rv_24", "rv_168", "har_rv_daily", "har_rv_weekly", "har_rv_monthly",
    "gk_vol_24", "yz_vol_24", "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct_b_20", "atr_14",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_asia", "session_london", "session_ny",
    "session_london_ny_overlap", "spread", "spread_mean_12", "spread_x_rv6",
    "xauh4_ret_4", "xauh4_zscore_4", "xauh4_ret_12", "xauh4_zscore_12",
    "xaud1_ret_5", "xaud1_zscore_5", "xaud1_ret_20", "xaud1_zscore_20",
]
EXTENDED_ONLY_COLS = [
    "dxy_ret_6", "dxy_zscore_6", "dxy_ret_24", "dxy_zscore_24",
    "ust_ret_6", "ust_zscore_6", "ust_ret_24", "ust_zscore_24",
]
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


def _load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _forward_return_target(xau_h1: pd.DataFrame, n: int = N_HORIZON) -> pd.Series:
    mid_close = (xau_h1["bid_close"] + xau_h1["ask_close"]) / 2
    fwd_time_ok = (xau_h1["time"].shift(-n) - xau_h1["time"]) == pd.Timedelta(hours=n)
    return np.log(mid_close.shift(-n) / mid_close).where(fwd_time_ok)


def load_modeling_dataset(profile: str = "core") -> pd.DataFrame:
    feat_files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=H1").glob("year=*/month=*.parquet"))
    feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    feat["time"] = pd.to_datetime(feat["time"])
    feat = feat.sort_values("time").reset_index(drop=True)

    xau = _load_symbol("XAUUSD", "H1")
    xau["target_fwd_ret_6"] = _forward_return_target(xau)
    feat = feat.merge(xau[["time", "target_fwd_ret_6"]], on="time", how="inner")

    cols = ["time"] + CORE_FEATURE_COLS + (EXTENDED_ONLY_COLS if profile == "extended" else []) + ["target_fwd_ret_6"]
    out = feat[cols].dropna().reset_index(drop=True)
    return out


def feature_cols(profile: str = "core") -> list[str]:
    return CORE_FEATURE_COLS + (EXTENDED_ONLY_COLS if profile == "extended" else [])
