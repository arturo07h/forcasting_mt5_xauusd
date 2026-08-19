"""Parses CSV output from mql5/exporters/XAUUSD_HistoryExporter.mq5 into partitioned Parquet.

No live MT5 connection happens here or anywhere in this project's Python side — the
MQL5 script is the only thing that talks to the broker. This module only reads files
already sitting on disk (copied over from the terminal's MQL5/Files folder).
"""
from pathlib import Path

import pandas as pd

_COLUMNS = [
    "time", "open", "high", "low", "close", "tick_volume", "broker_spread_points",
    "real_volume", "bid_open", "ask_open", "bid_close", "ask_close",
    "spread_mean_points", "spread_min_points", "spread_max_points", "tick_count",
]


def load_export_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M:%S")
    return df.sort_values("time").reset_index(drop=True)


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path, symbol: str) -> None:
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    out_dir.mkdir(parents=True, exist_ok=True)
    for (year, month), group in df.groupby(["year", "month"]):
        partition_dir = out_dir / f"symbol={symbol}" / f"year={year}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year", "month"]).to_parquet(
            partition_dir / f"month={month:02d}.parquet", index=False
        )


def parse_export(csv_path: Path, out_dir: Path, symbol: str) -> pd.DataFrame:
    df = load_export_csv(csv_path)
    write_partitioned_parquet(df, out_dir, symbol)
    return df
