"""Live inference service — bridges the MQL5 EA to the trained models via files, not a
live MT5↔Python connection (this Mac's MT5 is the native macOS app; see project memory
for why a direct connection isn't reachable). Reuses the exact feature code from
features/build_dataset_h1.py — nothing is reimplemented, which is the whole point of
this architecture over a native-MQL5 port.

Protocol (files live in the MT5 terminal's MQL5/Files/ folder — set MQL5_FILES_DIR
below to the real path before running):
  IN  (from EA, refreshed every new H1 bar close):
    xauusd_h1_bars.csv, xauusd_h4_bars.csv, xauusd_d1_bars.csv
      columns: time,open,high,low,close,spread_price
      (MT5's native CopyRates is a single, bid-convention OHLC series, not true bid+ask
      OHLC like the Dukascopy training data — spread_price, from each bar's own spread
      field, is used to approximate ask_* = native_* + spread_price for every OHLC point,
      not just the close, where the true value is known. A documented approximation, not
      a precision issue: spread is ~0.02% of price, so this barely moves any feature.)
  OUT (to EA):
    xauusd_h1_signal.json
      {"bar_time": "...", "action": "BUY"|"NONE", "sl_price": ..., "tp_price": ...,
       "computed_at": "...", "meta_proba": ..., "pred_tp_return": ...}

The EA only acts on a signal whose bar_time matches the bar it just closed — anything
older is treated as stale (service not running / lagging) and ignored, never traded on.

Run continuously: ./.venv/bin/python3 mql5/inference_service/live_inference.py
"""
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import DATA_RAW_DIR, PROJECT_ROOT
from features.fractional_diff import frac_diff_ffd
from features.volatility import realized_vol, har_rv, garman_klass, yang_zhang
from features.technical import rsi, macd, bollinger_pct_b, atr
from features.session import session_features
from features.higher_timeframe import trend_features, merge_asof_closed
from validation.dataset_h1 import feature_cols

# --- CONFIGURE THIS before running on the real machine ---
MQL5_FILES_DIR = Path.home() / "Library/Application Support/net.metaquotes.wine.metatrader5" / \
    "drive_c/Program Files/MetaTrader 5/MQL5/Files"
POLL_SECONDS = 20
FFD_D = 0.35
META_PROBA_THRESHOLD = 0.40
SL_ATR_MULT = 1.5
LOOKBACK_BARS_FOR_FEATURES = 3200  # FFD window (500, hits its max_size cap at d=0.35) + the 2160-bar
# price_ffd_zscore rolling window + margin — verified empirically: 2200 was NOT enough and silently
# produced a NaN price_ffd_zscore (caught by the missing-feature check, but don't regress this)

COLS = feature_cols("core")


def load_history_base() -> pd.DataFrame:
    """The deep Dukascopy H1 history already on disk — the live feed only ever
    supplies the newest few bars on top of this, so short/limited broker history
    (Fase 1's original concern) never matters for feature computation.
    """
    files = sorted((DATA_RAW_DIR / "symbol=XAUUSD" / "interval=H1").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True).tail(LOOKBACK_BARS_FOR_FEATURES * 2).reset_index(drop=True)


def load_mql5_bars(path: Path) -> pd.DataFrame:
    """Expands the EA's simple (open,high,low,close,spread_price) export into the same
    8-column bid/ask schema the Dukascopy history uses, via bid=native, ask=native+spread
    — see the module docstring for why this is a fine approximation.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[f"bid_{c}"] = df[c]
        df[f"ask_{c}"] = df[c] + df["spread_price"]
    return df.sort_values("time").reset_index(drop=True)


def merge_live_bars(base: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Appends any bars from the live (broker) feed that are newer than the historical
    base — the live feed is only ever a thin, recent extension of the deep history, not
    the primary source. Duplicate timestamps prefer the live value (freshest).
    """
    combined = pd.concat([base, live]).drop_duplicates(subset="time", keep="last")
    # reset_index again after tail() — tail() keeps the pre-tail positional labels, and any
    # code downstream that builds a fresh 0-indexed Series (like the FFD array) and assigns
    # it back would otherwise silently misalign and produce all-NaN
    return combined.sort_values("time").reset_index(drop=True).tail(LOOKBACK_BARS_FOR_FEATURES).reset_index(drop=True)


def compute_latest_features(xau_h1: pd.DataFrame, xau_h4: pd.DataFrame, xau_d1: pd.DataFrame) -> dict:
    xau_h1["mid_open"] = (xau_h1["bid_open"] + xau_h1["ask_open"]) / 2
    xau_h1["mid_high"] = (xau_h1["bid_high"] + xau_h1["ask_high"]) / 2
    xau_h1["mid_low"] = (xau_h1["bid_low"] + xau_h1["ask_low"]) / 2
    xau_h1["mid_close"] = (xau_h1["bid_close"] + xau_h1["ask_close"]) / 2
    xau_h1["spread"] = xau_h1["ask_close"] - xau_h1["bid_close"]

    time_diff_min = xau_h1["time"].diff().dt.total_seconds() / 60
    is_clean_step = time_diff_min == 60
    xau_h1["log_ret"] = np.log(xau_h1["mid_close"]).diff().where(is_clean_step)

    feat = pd.DataFrame({"time": xau_h1["time"]})

    ffd = frac_diff_ffd(xau_h1["mid_close"].values, FFD_D)
    pad = len(xau_h1) - len(ffd)
    raw_ffd = np.concatenate([np.full(pad, np.nan), ffd])
    # xau_h1's index is NOT reset after merge_live_bars' .tail() (it keeps the original
    # positional labels, e.g. 3200..6399) — a bare pd.Series(raw_ffd) gets a fresh 0-based
    # index instead, so assigning it into `feat` (indexed like xau_h1) would silently
    # align on nothing and produce all-NaN. Caught by the parity check against training
    # — this is exactly the kind of bug this bridge architecture exists to catch safely.
    ffd_series = pd.Series(raw_ffd, index=xau_h1.index)
    roll_mean = ffd_series.rolling(2160, min_periods=1944).mean()
    roll_std = ffd_series.rolling(2160, min_periods=1944).std()
    feat["price_ffd_zscore"] = (ffd_series - roll_mean) / roll_std

    feat = pd.concat([feat, realized_vol(xau_h1["log_ret"], windows=(6, 24, 168))], axis=1)
    feat = pd.concat([feat, har_rv(xau_h1["log_ret"], daily=24, weekly=24 * 5, monthly=24 * 22)], axis=1)
    feat["gk_vol_24"] = garman_klass(xau_h1["mid_open"], xau_h1["mid_high"], xau_h1["mid_low"], xau_h1["mid_close"], 24)
    feat["yz_vol_24"] = yang_zhang(xau_h1["mid_open"], xau_h1["mid_high"], xau_h1["mid_low"], xau_h1["mid_close"], 24)

    feat["rsi_14"] = rsi(xau_h1["mid_close"], 14)
    feat = pd.concat([feat, macd(xau_h1["mid_close"])], axis=1)
    feat["bb_pct_b_20"] = bollinger_pct_b(xau_h1["mid_close"], 20)
    feat["atr_14"] = atr(xau_h1["mid_high"], xau_h1["mid_low"], xau_h1["mid_close"], 14)

    feat = pd.concat([feat, session_features(xau_h1["time"])], axis=1)

    feat["spread"] = xau_h1["spread"]
    feat["spread_mean_12"] = xau_h1["spread"].rolling(12, min_periods=11).mean()
    feat["spread_x_rv6"] = xau_h1["spread"] * feat["rv_6"]

    h1_time = xau_h1["time"]
    xau_h4["close"] = (xau_h4["bid_close"] + xau_h4["ask_close"]) / 2
    h4_trend = trend_features(xau_h4, 14400, "xauh4", windows=(4, 12))
    feat = pd.concat([feat, merge_asof_closed(h1_time, h4_trend, ["xauh4_ret_4", "xauh4_zscore_4",
                                                                    "xauh4_ret_12", "xauh4_zscore_12"])], axis=1)

    xau_d1["close"] = (xau_d1["bid_close"] + xau_d1["ask_close"]) / 2
    d1_trend = trend_features(xau_d1, 86400, "xaud1", windows=(5, 20))
    feat = pd.concat([feat, merge_asof_closed(h1_time, d1_trend, ["xaud1_ret_5", "xaud1_zscore_5",
                                                                    "xaud1_ret_20", "xaud1_zscore_20"])], axis=1)

    feat["mid_close"] = xau_h1["mid_close"]
    feat["atr_14_raw"] = feat["atr_14"]
    return feat.iloc[-1].to_dict()


def main():
    print(f"Loading models and history base (this can take a few seconds)...")
    tp_model = lgb.Booster(model_file=str(PROJECT_ROOT / "models" / "checkpoints" / "xauusd_h1_tp_p90.txt"))
    meta_model = lgb.Booster(model_file=str(PROJECT_ROOT / "models" / "checkpoints" / "xauusd_h1_meta_label.txt"))
    history_base = load_history_base()
    last_processed_bar_time = None

    print(f"Watching {MQL5_FILES_DIR} — polling every {POLL_SECONDS}s...")
    while True:
        bars_path = MQL5_FILES_DIR / "xauusd_h1_bars.csv"
        h4_path = MQL5_FILES_DIR / "xauusd_h4_bars.csv"
        d1_path = MQL5_FILES_DIR / "xauusd_d1_bars.csv"
        if not (bars_path.exists() and h4_path.exists() and d1_path.exists()):
            print("waiting for EA to write bar files...")
            time.sleep(POLL_SECONDS)
            continue

        try:
            live_h1 = load_mql5_bars(bars_path)
            live_h4 = load_mql5_bars(h4_path)
            live_d1 = load_mql5_bars(d1_path)
        except Exception as e:
            print(f"error reading bar files (mid-write?): {e} — retrying")
            time.sleep(POLL_SECONDS)
            continue

        latest_bar_time = live_h1["time"].iloc[-1]
        if latest_bar_time == last_processed_bar_time:
            time.sleep(POLL_SECONDS)
            continue

        print(f"New bar detected: {latest_bar_time}")
        xau_h1 = merge_live_bars(history_base, live_h1)
        row = compute_latest_features(xau_h1, live_h4, live_d1)

        feature_values = [row.get(c) for c in COLS]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in feature_values):
            missing = [c for c, v in zip(COLS, feature_values) if v is None or (isinstance(v, float) and np.isnan(v))]
            print(f"WARNING: missing features {missing}, skipping this bar (no signal written)")
            last_processed_bar_time = latest_bar_time
            time.sleep(POLL_SECONDS)
            continue

        X = pd.DataFrame([feature_values], columns=COLS)
        pred_tp_return = float(tp_model.predict(X)[0])
        meta_proba = float(meta_model.predict(X)[0])

        action = "NONE"
        sl_price, tp_price = None, None
        if meta_proba >= META_PROBA_THRESHOLD and pred_tp_return > 0:
            action = "BUY"
            entry = row["mid_close"]
            sl_price = entry - SL_ATR_MULT * row["atr_14_raw"]
            tp_price = entry * (1 + pred_tp_return)

        signal = {
            "bar_time": str(latest_bar_time), "action": action,
            "sl_price": sl_price, "tp_price": tp_price,
            "meta_proba": meta_proba, "pred_tp_return": pred_tp_return,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        signal_path = MQL5_FILES_DIR / "xauusd_h1_signal.json"
        signal_path.write_text(json.dumps(signal, indent=2))
        print(f"  -> {action}  meta_proba={meta_proba:.3f}  pred_tp_return={pred_tp_return:.4f}")

        last_processed_bar_time = latest_bar_time
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
