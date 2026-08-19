"""Assembles the modeling dataset: processed features (features/build_dataset.py output)
joined with the N=12-candle forward-return target (same gap-aware definition as the EDA:
notebooks/eda_xauusd_m5.py section 8 — a window is only labeled if it doesn't cross a
data gap). Two feature profiles are offered, per the trade-off documented in
notebooks/eda_summary.json / project memory:
  - "core": no DXY/USTBOND, usable across the full 2005-2026 history
  - "extended": includes DXY/USTBOND, usable only from ~2020 onward
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR

N_HORIZON = 12

PRICE_FFD_ZSCORE_WINDOW = 25920  # ~90 days of M5 bars — long enough to keep meaningful
# trend memory (the whole point of using FFD over plain returns), short enough that the
# feature can never drift 10+ std outside its own recent range the way the raw FFD level
# did across gold's decade-scale price appreciation (see project memory: fold 5 test
# z-scores under fixed train-period standardization reached 15.5)

CORE_FEATURE_COLS = [
    "price_ffd_zscore", "rv_12", "rv_48", "rv_288", "har_rv_daily", "har_rv_weekly", "har_rv_monthly",
    "gk_vol_48", "yz_vol_48", "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct_b_20", "atr_14",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_asia", "session_london", "session_ny",
    "session_london_ny_overlap", "spread", "spread_mean_12", "spread_x_rv12",
    "xauh1_ret_4", "xauh1_zscore_4", "xauh1_ret_12", "xauh1_zscore_12",
    "xauh4_ret_4", "xauh4_zscore_4", "xauh4_ret_12", "xauh4_zscore_12",
]
EXTENDED_ONLY_COLS = [
    "dxy_ret_12", "dxy_zscore_12", "dxy_ret_48", "dxy_zscore_48",
    "ust_ret_12", "ust_zscore_12", "ust_ret_48", "ust_zscore_48",
]
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]


def _load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _forward_return_target(xau_m5: pd.DataFrame, n: int = N_HORIZON) -> pd.Series:
    mid_close = (xau_m5["bid_close"] + xau_m5["ask_close"]) / 2
    fwd_time_ok = (xau_m5["time"].shift(-n) - xau_m5["time"]) == pd.Timedelta(minutes=5 * n)
    return np.log(mid_close.shift(-n) / mid_close).where(fwd_time_ok)


def load_modeling_dataset(profile: str = "core") -> pd.DataFrame:
    feat_files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5").glob("year=*/month=*.parquet"))
    feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    feat["time"] = pd.to_datetime(feat["time"])
    feat = feat.sort_values("time").reset_index(drop=True)

    xau = _load_symbol("XAUUSD", "M5")
    xau["target_fwd_ret_12"] = _forward_return_target(xau)
    feat = feat.merge(xau[["time", "target_fwd_ret_12"]], on="time", how="inner")

    # rolling (not fixed-period) z-score: price_ffd's raw level drifts with the price
    # trend over years (gold ~$450 -> ~$4000+ across this dataset) even though it passes
    # ADF — "stationary" doesn't mean "bounded within any practical modeling window,"
    # and a fixed train-period mean/std leaves later test folds many std devs out of range
    mp = int(PRICE_FFD_ZSCORE_WINDOW * 0.9)
    roll_mean = feat["price_ffd"].rolling(PRICE_FFD_ZSCORE_WINDOW, min_periods=mp).mean()
    roll_std = feat["price_ffd"].rolling(PRICE_FFD_ZSCORE_WINDOW, min_periods=mp).std()
    feat["price_ffd_zscore"] = (feat["price_ffd"] - roll_mean) / roll_std

    cols = ["time"] + CORE_FEATURE_COLS + (EXTENDED_ONLY_COLS if profile == "extended" else []) + ["target_fwd_ret_12"]
    out = feat[cols].dropna().reset_index(drop=True)
    return out


def feature_cols(profile: str = "core") -> list[str]:
    return CORE_FEATURE_COLS + (EXTENDED_ONLY_COLS if profile == "extended" else [])
