"""Applies triple-barrier labeling to the full XAUUSD M5 history.

Barrier rule used here (a placeholder until the quantile-regression model exists):
  SL = 1.5 x ATR(14)                                    — "well-defined, not arbitrary"
  TP = entry x |p95_unconditional| x (har_rv_daily / expanding_median(har_rv_daily))
       — EDA's empirical p95 forward-return magnitude, scaled by how the current
       volatility regime compares to everything seen so far (expanding, not global,
       median — so this doesn't use future information at any point t)

This deliberately encodes "eat the whole pie": TP targets the p95 magnitude while SL
sits at a much tighter ATR-based distance, not a matched quantile — consistent with
the user's stated risk philosophy, not a mistake to be smoothed into symmetry.

Once the real quantile model exists, its predicted p95 (or whichever TP quantile wins
empirically) replaces the placeholder TP rule directly; the barrier-scan mechanics
(triple_barrier_scan) don't change.

Run: ./.venv/bin/python3 labeling/build_labels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
from labeling.triple_barrier import triple_barrier_scan

MAX_HORIZON = 48          # vertical barrier: 48 candles (~4h) — generous vs the 12-candle (~1h) EDA target,
                           # since a real trade may need more room to reach TP/SL than a fixed-horizon return check
SL_ATR_MULT = 1.5
TP_P95_UNCOND = 0.0032265182792715753   # from notebooks/eda_summary.json, forward_return_target.unconditional_quantiles["0.95"]


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def load_processed_features() -> pd.DataFrame:
    files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build():
    print("Loading raw OHLC and processed features...")
    xau = load_symbol("XAUUSD", "M5")
    xau["mid_open"] = (xau["bid_open"] + xau["ask_open"]) / 2
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    feat = load_processed_features()[["time", "atr_14", "har_rv_daily"]]
    df = xau.merge(feat, on="time", how="inner").reset_index(drop=True)
    print(f"  {len(df):,} bars after joining features")

    print("Building validity mask (no bad data gaps inside the horizon window)...")
    time_diff_min = df["time"].diff().dt.total_seconds() / 60
    is_anomalous_gap = (time_diff_min > 5) & (time_diff_min <= 1000)
    # a label starting at t is invalid if any bar in (t, t+MAX_HORIZON] arrived via a bad gap
    gap_shifted = is_anomalous_gap.shift(-1)  # position t now holds bar t+1's gap status
    bad_gap_ahead = gap_shifted[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].fillna(0).astype(bool)
    not_near_end = np.arange(len(df)) < (len(df) - MAX_HORIZON - 1)
    has_features = df["atr_14"].notna() & df["har_rv_daily"].notna()
    valid = (~bad_gap_ahead) & not_near_end & has_features
    print(f"  {valid.sum():,} valid starting bars ({valid.mean()*100:.1f}%)")

    print("Computing barrier distances (SL=ATR-based, TP=expanding-vol-scaled p95)...")
    vol_regime_ratio = df["har_rv_daily"] / df["har_rv_daily"].expanding(min_periods=288).median()
    sl_dist = SL_ATR_MULT * df["atr_14"]
    tp_dist = df["mid_close"] * TP_P95_UNCOND * vol_regime_ratio
    valid = valid & vol_regime_ratio.notna()

    print(f"Running triple-barrier scan (max_horizon={MAX_HORIZON} bars)...")
    label, ret, time_to_hit = triple_barrier_scan(
        df["mid_high"].values, df["mid_low"].values, df["mid_close"].values,
        tp_dist.values, sl_dist.values, MAX_HORIZON, valid.values,
    )
    df["label"] = label
    df["realized_ret"] = ret
    df["time_to_hit"] = time_to_hit
    df["sl_dist"] = sl_dist
    df["tp_dist"] = tp_dist

    labeled = df.loc[valid].copy()
    print(f"\n{len(labeled):,} labeled bars")
    print("\nLabel distribution:")
    print(labeled["label"].value_counts(normalize=True).rename({1: "TP", -1: "SL", 0: "timeout"}))

    print("\nTime-to-hit (bars) by label:")
    print(labeled.groupby("label")["time_to_hit"].describe()[["mean", "50%", "max"]])

    print("\nRealized return by label:")
    print(labeled.groupby("label")["realized_ret"].describe()[["mean", "std", "min", "max"]])

    rr = (labeled["tp_dist"] / labeled["sl_dist"]).describe()
    print("\nTP:SL distance ratio (the built-in reward:risk of this barrier rule):")
    print(rr[["mean", "50%", "min", "max"]])

    out_dir = PROJECT_ROOT / "data" / "processed" / "labels" / "symbol=XAUUSD"
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
