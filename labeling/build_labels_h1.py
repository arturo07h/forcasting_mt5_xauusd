"""H1 counterpart of build_labels.py. Same placeholder barrier rule (SL=ATR-based,
TP=unconditional-p95-magnitude scaled by current/expanding vol regime), re-parameterized:
MAX_HORIZON=24 H1 bars (~1 day, room for a 6h-horizon-informed trade to resolve),
TP_P95_UNCOND from this dataset's own target distribution (0.854%, not the M5 value).

Run: ./.venv/bin/python3 labeling/build_labels_h1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
from labeling.triple_barrier import triple_barrier_scan

MAX_HORIZON = 24  # H1 bars (~1 day)
SL_ATR_MULT = 1.5
TP_P95_UNCOND = 0.008539671320771882  # forward_return_target p95, H1 N=6h — see validation/dataset_h1.py


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def load_processed_features() -> pd.DataFrame:
    files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=H1").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build():
    print("Loading raw OHLC and processed features...")
    xau = load_symbol("XAUUSD", "H1")
    xau["mid_open"] = (xau["bid_open"] + xau["ask_open"]) / 2
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    feat = load_processed_features()[["time", "atr_14", "har_rv_daily"]]
    df = xau.merge(feat, on="time", how="inner").reset_index(drop=True)
    print(f"  {len(df):,} bars after joining features")

    time_diff_min = df["time"].diff().dt.total_seconds() / 60
    is_anomalous_gap = (time_diff_min > 60) & (time_diff_min <= 1000)
    gap_shifted = is_anomalous_gap.shift(-1)
    bad_gap_ahead = gap_shifted[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].fillna(0).astype(bool)
    not_near_end = np.arange(len(df)) < (len(df) - MAX_HORIZON - 1)
    has_features = df["atr_14"].notna() & df["har_rv_daily"].notna()

    vol_regime_ratio = df["har_rv_daily"] / df["har_rv_daily"].expanding(min_periods=24).median()
    sl_dist = SL_ATR_MULT * df["atr_14"]
    tp_dist = df["mid_close"] * TP_P95_UNCOND * vol_regime_ratio
    valid = (~bad_gap_ahead) & not_near_end & has_features & vol_regime_ratio.notna()
    print(f"  {valid.sum():,} valid starting bars ({valid.mean()*100:.1f}%)")

    label, ret, time_to_hit = triple_barrier_scan(
        df["mid_high"].values, df["mid_low"].values, df["mid_close"].values,
        tp_dist.values, sl_dist.values, MAX_HORIZON, valid.values,
    )
    df["label"], df["realized_ret"], df["time_to_hit"] = label, ret, time_to_hit
    df["sl_dist"], df["tp_dist"] = sl_dist, tp_dist

    labeled = df.loc[valid].copy()
    print(f"\n{len(labeled):,} labeled bars")
    print(labeled["label"].value_counts(normalize=True).rename({1: "TP", -1: "SL", 0: "timeout"}))

    out_dir = PROJECT_ROOT / "data" / "processed" / "labels_h1" / "symbol=XAUUSD"
    out_cols = ["time", "mid_close", "label", "realized_ret", "time_to_hit", "sl_dist", "tp_dist"]
    labeled["year"] = labeled["time"].dt.year
    labeled["month"] = labeled["time"].dt.month
    for (y, m), group in labeled.groupby(["year", "month"]):
        partition_dir = out_dir / f"year={y}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group[out_cols].to_parquet(partition_dir / f"month={m:02d}.parquet", index=False)
    print(f"\nSaved to {out_dir}")
    return labeled


if __name__ == "__main__":
    build()
