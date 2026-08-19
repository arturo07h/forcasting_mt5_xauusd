"""Aggregates already-downloaded M5 data up to H1 (DXY, USTBOND — no need to re-hit
Dukascopy) and XAUUSD H1 up to D1 (higher-timeframe context for the H1 pipeline).
Saves into the same data/raw/ partition layout, interval=H1 / interval=D1, so the H1
pipeline can load them exactly like the natively-downloaded XAUUSD H1/H4.

Run: ./.venv/bin/python3 ingestion/aggregate_to_h1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DATA_RAW_DIR


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def aggregate_ohlc_single_side(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.set_index("time")
    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()
    return out.reset_index()


def aggregate_ohlc_bid_ask(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.set_index("time")
    cols = {}
    for side in ["bid", "ask"]:
        cols[f"{side}_open"] = df[f"{side}_open"].resample(rule).first()
        cols[f"{side}_high"] = df[f"{side}_high"].resample(rule).max()
        cols[f"{side}_low"] = df[f"{side}_low"].resample(rule).min()
        cols[f"{side}_close"] = df[f"{side}_close"].resample(rule).last()
    out = pd.DataFrame(cols).dropna()
    return out.reset_index()


def save_partitioned(df: pd.DataFrame, symbol: str, interval: str):
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    out_dir = DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}"
    for (y, m), group in df.groupby(["year", "month"]):
        partition_dir = out_dir / f"year={y}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year", "month"]).to_parquet(partition_dir / f"month={m:02d}.parquet", index=False)
    print(f"saved {symbol}/{interval}: {len(df):,} bars -> {out_dir}")


def main():
    print("DXY M5 -> H1...")
    dxy_m5 = load_symbol("DXY", "M5")
    dxy_h1 = aggregate_ohlc_single_side(dxy_m5, "1h")
    save_partitioned(dxy_h1, "DXY", "H1")

    print("USTBOND M5 -> H1...")
    ust_m5 = load_symbol("USTBOND", "M5")
    ust_h1 = aggregate_ohlc_single_side(ust_m5, "1h")
    save_partitioned(ust_h1, "USTBOND", "H1")

    # Rebuilt from our own already gap-audited M5 data rather than the native H1
    # download, for a single consistent data lineage — checked by hand, this turned
    # out NOT to fix the ~200/year hour-scale gaps seen in 2016+ (same rate either way):
    # those are genuine M5-level data gaps (median ~10min, tail to 725min — see the
    # EDA), and a subset of them are wide enough to span a full empty hour. At H1 the
    # same absolute count of gap events lands on ~12x fewer total bars than at M5, so
    # each one invalidates proportionally more triple-barrier windows — a real
    # characteristic of the coarser timeframe, not a bug to keep chasing.
    print("XAUUSD M5 -> H1 (for a single consistent data lineage)...")
    xau_m5 = load_symbol("XAUUSD", "M5")
    xau_h1 = aggregate_ohlc_bid_ask(xau_m5, "1h")
    save_partitioned(xau_h1, "XAUUSD", "H1")

    print("XAUUSD H1 (M5-derived) -> D1...")
    xau_d1 = aggregate_ohlc_bid_ask(xau_h1, "1D")
    save_partitioned(xau_d1, "XAUUSD", "D1")


if __name__ == "__main__":
    main()
