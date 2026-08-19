"""Pulls historical OHLC (bid+ask) from Dukascopy via the pure-Python `dukascopy-python`
client. This is the primary research/training data source. The MT5 exporter in
mql5/exporters/ pulls the broker-specific feed separately, used later to validate
execution realism (spread, fills) before deployment — the two are not meant to be mixed
into one dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import dukascopy_python as dk
import pandas as pd

INTERVAL_LABELS = {
    dk.INTERVAL_MIN_5: "M5",
    dk.INTERVAL_HOUR_1: "H1",
    dk.INTERVAL_HOUR_4: "H4",
    dk.INTERVAL_DAY_1: "D1",
}


@dataclass
class SymbolSpec:
    label: str        # our internal name, e.g. "XAUUSD"
    instrument: str    # dukascopy instrument code, e.g. "XAU/USD"


def fetch_bid_ask(instrument: str, interval: str, start: datetime, end: datetime,
                   max_retries: int = 7) -> pd.DataFrame:
    """Bid and ask kept as separate columns — never averaged into a mid price here.
    Spread is derived downstream from the two, not baked in and thrown away.
    """
    bid = dk.fetch(instrument, interval, dk.OFFER_SIDE_BID, start, end, max_retries=max_retries)
    ask = dk.fetch(instrument, interval, dk.OFFER_SIDE_ASK, start, end, max_retries=max_retries)
    bid = bid.add_prefix("bid_")
    ask = ask.add_prefix("ask_")
    df = bid.join(ask, how="inner")
    dropped = max(len(bid), len(ask)) - len(df)
    if dropped > 0:
        print(f"WARNING: {dropped} bars dropped (bid/ask row mismatch) for {instrument} {interval}")
    return df


def fetch_single_side(instrument: str, interval: str, start: datetime, end: datetime,
                       side: str = dk.OFFER_SIDE_BID, max_retries: int = 7) -> pd.DataFrame:
    return dk.fetch(instrument, interval, side, start, end, max_retries=max_retries)


def _month_chunks(start: datetime, end: datetime):
    """Upper bound of each chunk is exclusive (nudged back 1s) — dukascopy_python's fetch()
    includes a bar exactly at `end`, so without this the boundary bar is duplicated between
    two adjacent months' output files.
    """
    cur = datetime(start.year, start.month, 1)
    while cur < end:
        nxt = datetime(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt, end) - timedelta(seconds=1)
        yield chunk_start, chunk_end
        cur = nxt


def download_range(
    spec: SymbolSpec,
    interval: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
    with_spread: bool = True,
    skip_existing: bool = True,
) -> None:
    """Downloads month by month into partitioned Parquet, skipping months already on disk
    so an interrupted run can resume without re-pulling everything.
    """
    interval_label = INTERVAL_LABELS[interval]
    for chunk_start, chunk_end in _month_chunks(start, end):
        partition_dir = out_dir / f"symbol={spec.label}" / f"interval={interval_label}" / f"year={chunk_start.year}"
        partition_path = partition_dir / f"month={chunk_start.month:02d}.parquet"

        if skip_existing and partition_path.exists():
            print(f"skip {spec.label} {interval_label} {chunk_start:%Y-%m}: already on disk")
            continue

        try:
            if with_spread:
                df = fetch_bid_ask(spec.instrument, interval, chunk_start, chunk_end)
            else:
                df = fetch_single_side(spec.instrument, interval, chunk_start, chunk_end)
        except Exception as e:
            print(f"ERROR {spec.label} {interval_label} {chunk_start:%Y-%m}: {e}")
            continue

        if df.empty:
            print(f"empty {spec.label} {interval_label} {chunk_start:%Y-%m} — likely before available history")
            continue

        # month-boundary bar can come back in both adjacent chunk requests
        df = df[~df.index.duplicated(keep="first")]

        partition_dir.mkdir(parents=True, exist_ok=True)
        df.reset_index().rename(columns={"timestamp": "time"}).to_parquet(partition_path, index=False)
        print(f"saved {spec.label} {interval_label} {chunk_start:%Y-%m}: {len(df)} bars")
