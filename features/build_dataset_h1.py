"""H1 counterpart of build_dataset.py — same feature modules, window parameters
rescaled for the coarser timeframe (not just reusing the M5 bar-counts, which would
mean very different real-world durations). Built after the M5 backtest showed spread
eats ~26% of the risk unit (1.5xATR) at M5 vs ~7% at H1 — see project memory / the
final report for the diagnostic that motivated this rebuild.

Run: ./.venv/bin/python3 features/build_dataset_h1.py
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

FFD_D = 0.35  # H1-specific: min-d passing ADF was 0.30, used 0.35 for a comfortable margin (see chat/memory)


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build():
    print("Loading XAUUSD H1...")
    xau = load_symbol("XAUUSD", "H1")
    xau["mid_open"] = (xau["bid_open"] + xau["ask_open"]) / 2
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2
    xau["spread"] = xau["ask_close"] - xau["bid_close"]

    time_diff_min = xau["time"].diff().dt.total_seconds() / 60
    is_clean_step = time_diff_min == 60
    xau["log_ret"] = np.log(xau["mid_close"]).diff().where(is_clean_step)

    feat = pd.DataFrame({"time": xau["time"]})

    print("Fractional differentiation...")
    ffd = frac_diff_ffd(xau["mid_close"].values, FFD_D)
    pad = len(xau) - len(ffd)
    raw_ffd = np.concatenate([np.full(pad, np.nan), ffd])
    # rolling z-score (not fixed-period) — same lesson as the M5 pipeline: a slowly
    # trending level feature needs this to stay in-range for a neural net, and it's
    # harmless for tree models. Window ~90 days in H1 bars.
    ffd_series = pd.Series(raw_ffd)
    roll_mean = ffd_series.rolling(2160, min_periods=1944).mean()
    roll_std = ffd_series.rolling(2160, min_periods=1944).std()
    feat["price_ffd_zscore"] = (ffd_series - roll_mean) / roll_std

    print("Volatility features...")
    feat = pd.concat([feat, realized_vol(xau["log_ret"], windows=(6, 24, 168))], axis=1)
    feat = pd.concat([feat, har_rv(xau["log_ret"], daily=24, weekly=24 * 5, monthly=24 * 22)], axis=1)
    feat["gk_vol_24"] = garman_klass(xau["mid_open"], xau["mid_high"], xau["mid_low"], xau["mid_close"], 24)
    feat["yz_vol_24"] = yang_zhang(xau["mid_open"], xau["mid_high"], xau["mid_low"], xau["mid_close"], 24)

    print("Technical indicators...")
    feat["rsi_14"] = rsi(xau["mid_close"], 14)
    feat = pd.concat([feat, macd(xau["mid_close"])], axis=1)
    feat["bb_pct_b_20"] = bollinger_pct_b(xau["mid_close"], 20)
    feat["atr_14"] = atr(xau["mid_high"], xau["mid_low"], xau["mid_close"], 14)

    print("Session features...")
    feat = pd.concat([feat, session_features(xau["time"])], axis=1)

    print("Spread features...")
    feat["spread"] = xau["spread"]
    feat["spread_mean_12"] = xau["spread"].rolling(12, min_periods=11).mean()
    feat["spread_x_rv6"] = xau["spread"] * feat["rv_6"]

    print("Higher-timeframe context (XAUUSD H4/D1, DXY, USTBOND)...")
    h1_time = xau["time"]

    xau_h4 = load_symbol("XAUUSD", "H4")
    xau_h4["close"] = (xau_h4["bid_close"] + xau_h4["ask_close"]) / 2
    h4_trend = trend_features(xau_h4, 14400, "xauh4", windows=(4, 12))
    feat = pd.concat([feat, merge_asof_closed(h1_time, h4_trend, ["xauh4_ret_4", "xauh4_zscore_4",
                                                                    "xauh4_ret_12", "xauh4_zscore_12"])], axis=1)

    xau_d1 = load_symbol("XAUUSD", "D1")
    xau_d1["close"] = (xau_d1["bid_close"] + xau_d1["ask_close"]) / 2
    d1_trend = trend_features(xau_d1, 86400, "xaud1", windows=(5, 20))
    feat = pd.concat([feat, merge_asof_closed(h1_time, d1_trend, ["xaud1_ret_5", "xaud1_zscore_5",
                                                                    "xaud1_ret_20", "xaud1_zscore_20"])], axis=1)

    dxy = load_symbol("DXY", "H1")
    dxy_trend = trend_features(dxy, 3600, "dxy", windows=(6, 24))
    feat = pd.concat([feat, merge_asof_closed(h1_time, dxy_trend, ["dxy_ret_6", "dxy_zscore_6",
                                                                     "dxy_ret_24", "dxy_zscore_24"])], axis=1)

    ustbond = load_symbol("USTBOND", "H1")
    ust_trend = trend_features(ustbond, 3600, "ust", windows=(6, 24))
    feat = pd.concat([feat, merge_asof_closed(h1_time, ust_trend, ["ust_ret_6", "ust_zscore_6",
                                                                     "ust_ret_24", "ust_zscore_24"])], axis=1)

    feature_cols = [c for c in feat.columns if c != "time"]
    print(f"\n{len(feature_cols)} feature columns, {len(feat):,} rows before warmup trim")

    na_pct = feat[feature_cols].isna().mean().sort_values(ascending=False)
    print("\nNaN %% by column (top 10):")
    print(na_pct.head(10))

    complete = feat.dropna(subset=feature_cols)
    print(f"\n{len(complete):,} rows with all features present ({len(complete)/len(feat)*100:.1f}%)")
    print(f"usable range: {complete['time'].min()} -> {complete['time'].max()}")

    feat["year"] = feat["time"].dt.year
    feat["month"] = feat["time"].dt.month
    out_dir = DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=H1"
    for (y, m), group in feat.groupby(["year", "month"]):
        partition_dir = out_dir / f"year={y}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year", "month"]).to_parquet(partition_dir / f"month={m:02d}.parquet", index=False)
    print(f"\nSaved to {out_dir}")

    return feat, feature_cols


if __name__ == "__main__":
    build()
