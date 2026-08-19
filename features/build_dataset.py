"""Builds the full XAUUSD M5 feature matrix from data/raw/ and saves it to
data/processed/. Run: ./.venv/bin/python3 features/build_dataset.py

Every feature here is strictly causal (rolling/expanding windows only, no centered
windows, no future data) and gap-aware (built on the gap-masked log-return series from
the EDA fix — a window spanning a data hole yields NaN, never a spurious value).
Labeling (triple-barrier, the actual quantile-regression target) is a separate stage —
this module only produces X, not y.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR
from features.fractional_diff import frac_diff_ffd
from features.volatility import realized_vol, har_rv, garman_klass, yang_zhang
from features.technical import rsi, macd, bollinger_pct_b, atr
from features.session import session_features
from features.higher_timeframe import trend_features, merge_asof_closed

FFD_D = 0.4  # from EDA: minimal d passing ADF was 0.35-0.4; re-validate per training fold later


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build():
    print("Loading XAUUSD M5...")
    xau = load_symbol("XAUUSD", "M5")
    xau["mid_open"] = (xau["bid_open"] + xau["ask_open"]) / 2
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2
    xau["spread"] = xau["ask_close"] - xau["bid_close"]

    time_diff_min = xau["time"].diff().dt.total_seconds() / 60
    is_clean_step = time_diff_min == 5
    xau["log_ret"] = np.log(xau["mid_close"]).diff().where(is_clean_step)

    feat = pd.DataFrame({"time": xau["time"]})

    print("Fractional differentiation...")
    ffd = frac_diff_ffd(xau["mid_close"].values, FFD_D)
    pad = len(xau) - len(ffd)
    feat["price_ffd"] = np.concatenate([np.full(pad, np.nan), ffd])

    print("Volatility features...")
    feat = pd.concat([feat, realized_vol(xau["log_ret"], windows=(12, 48, 288))], axis=1)
    feat = pd.concat([feat, har_rv(xau["log_ret"])], axis=1)
    feat["gk_vol_48"] = garman_klass(xau["mid_open"], xau["mid_high"], xau["mid_low"], xau["mid_close"], 48)
    feat["yz_vol_48"] = yang_zhang(xau["mid_open"], xau["mid_high"], xau["mid_low"], xau["mid_close"], 48)

    print("Technical indicators...")
    feat["rsi_14"] = rsi(xau["mid_close"], 14)
    feat = pd.concat([feat, macd(xau["mid_close"])], axis=1)
    feat["bb_pct_b_20"] = bollinger_pct_b(xau["mid_close"], 20)
    feat["atr_14"] = atr(xau["mid_high"], xau["mid_low"], xau["mid_close"], 14)

    print("Session features...")
    feat = pd.concat([feat, session_features(xau["time"])], axis=1)

    print("Spread features...")
    feat["spread"] = xau["spread"]
    feat["spread_mean_12"] = xau["spread"].rolling(12).mean()
    feat["spread_x_rv12"] = xau["spread"] * feat["rv_12"]

    print("Higher-timeframe context (XAUUSD H1/H4, DXY, USTBOND)...")
    m5_time = xau["time"]

    xau_h1 = load_symbol("XAUUSD", "H1")
    xau_h1["close"] = (xau_h1["bid_close"] + xau_h1["ask_close"]) / 2
    h1_trend = trend_features(xau_h1, 3600, "xauh1", windows=(4, 12))
    feat = pd.concat([feat, merge_asof_closed(m5_time, h1_trend, ["xauh1_ret_4", "xauh1_zscore_4",
                                                                    "xauh1_ret_12", "xauh1_zscore_12"])], axis=1)

    xau_h4 = load_symbol("XAUUSD", "H4")
    xau_h4["close"] = (xau_h4["bid_close"] + xau_h4["ask_close"]) / 2
    h4_trend = trend_features(xau_h4, 14400, "xauh4", windows=(4, 12))
    feat = pd.concat([feat, merge_asof_closed(m5_time, h4_trend, ["xauh4_ret_4", "xauh4_zscore_4",
                                                                    "xauh4_ret_12", "xauh4_zscore_12"])], axis=1)

    dxy = load_symbol("DXY", "M5")
    dxy_trend = trend_features(dxy, 300, "dxy", windows=(12, 48))
    feat = pd.concat([feat, merge_asof_closed(m5_time, dxy_trend, ["dxy_ret_12", "dxy_zscore_12",
                                                                     "dxy_ret_48", "dxy_zscore_48"])], axis=1)

    ustbond = load_symbol("USTBOND", "M5")
    ust_trend = trend_features(ustbond, 300, "ust", windows=(12, 48))
    feat = pd.concat([feat, merge_asof_closed(m5_time, ust_trend, ["ust_ret_12", "ust_zscore_12",
                                                                     "ust_ret_48", "ust_zscore_48"])], axis=1)

    feature_cols = [c for c in feat.columns if c != "time"]
    print(f"\n{len(feature_cols)} feature columns, {len(feat):,} rows before warmup trim")

    na_pct = feat[feature_cols].isna().mean().sort_values(ascending=False)
    print("\nNaN % by column (expected at the start of the series from rolling warmup):")
    print(na_pct.head(15))

    complete = feat.dropna(subset=feature_cols)
    print(f"\n{len(complete):,} rows with all features present ({len(complete)/len(feat)*100:.1f}%)")
    print(f"usable range: {complete['time'].min()} -> {complete['time'].max()}")

    feat["year"] = feat["time"].dt.year
    feat["month"] = feat["time"].dt.month
    out_dir = DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5"
    for (y, m), group in feat.groupby(["year", "month"]):
        partition_dir = out_dir / f"year={y}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year", "month"]).to_parquet(partition_dir / f"month={m:02d}.parquet", index=False)
    print(f"\nSaved to {out_dir}")

    return feat, feature_cols


if __name__ == "__main__":
    build()
