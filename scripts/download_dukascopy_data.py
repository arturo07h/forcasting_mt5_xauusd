"""Downloads the full research dataset for this project from Dukascopy.

Per-symbol start dates were chosen empirically (checked in a REPL before this run) —
Dukascopy's free feed doesn't have uniform depth across instruments:
  - XAUUSD M5/H1/H4: reliable from 2005-01
  - DXY (DOLLAR.IDX/USD) M5: reliable from 2018-01
  - USTBOND.TR/USD M5: reliable from 2020-01 (rate-direction proxy; no direct
    UST yield series is available on this feed)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dukascopy_python import INTERVAL_MIN_5, INTERVAL_HOUR_1, INTERVAL_HOUR_4
from dukascopy_python.instruments import (
    INSTRUMENT_FX_METALS_XAU_USD,
    INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD,
    INSTRUMENT_BND_CFD_USTBOND_TR_USD,
)

from config.settings import DATA_RAW_DIR
from ingestion.dukascopy_client import SymbolSpec, download_range

NOW = datetime.now(timezone.utc).replace(tzinfo=None)

XAUUSD = SymbolSpec(label="XAUUSD", instrument=INSTRUMENT_FX_METALS_XAU_USD)
DXY = SymbolSpec(label="DXY", instrument=INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD)
USTBOND = SymbolSpec(label="USTBOND", instrument=INSTRUMENT_BND_CFD_USTBOND_TR_USD)


def main():
    jobs = [
        (XAUUSD, INTERVAL_MIN_5, datetime(2005, 1, 1), NOW, True),
        (XAUUSD, INTERVAL_HOUR_1, datetime(2005, 1, 1), NOW, True),
        (XAUUSD, INTERVAL_HOUR_4, datetime(2005, 1, 1), NOW, True),
        (DXY, INTERVAL_MIN_5, datetime(2018, 1, 1), NOW, False),
        (USTBOND, INTERVAL_MIN_5, datetime(2020, 1, 1), NOW, False),
    ]
    for spec, interval, start, end, with_spread in jobs:
        print(f"=== {spec.label} {interval} {start:%Y-%m} -> {end:%Y-%m} ===", flush=True)
        download_range(spec, interval, start, end, DATA_RAW_DIR, with_spread=with_spread)


if __name__ == "__main__":
    main()
