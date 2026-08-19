"""OHLC volatility estimators and HAR-RV components, all rolling/causal (no centered
windows, no look-ahead). All operate on the gap-masked clean log-return series so a
window spanning a data gap yields NaN instead of a spurious spike — see the EDA note
on the 2005-2012 gap issue in notebooks/eda_xauusd_m5.py.
"""
import numpy as np
import pandas as pd

_LN2 = np.log(2)


def _min_periods(window: int) -> int:
    """A window this long will almost always contain a few gap-masked NaNs (a week
    always spans a weekend close) — requiring a full window with zero NaN would make
    every weekly/monthly estimate NaN. 90% coverage tolerates the routine weekend gaps
    (and the rarer intraday ones) without silently accepting a mostly-empty window.
    """
    return max(1, int(window * 0.9))


def realized_vol(log_ret: pd.Series, windows=(12, 48, 288)) -> pd.DataFrame:
    """Simple rolling std of returns — windows default to ~1h/4h/1day in M5 bars."""
    out = {}
    for w in windows:
        out[f"rv_{w}"] = log_ret.rolling(w, min_periods=_min_periods(w)).std()
    return pd.DataFrame(out, index=log_ret.index)


def har_rv(log_ret: pd.Series, daily=288, weekly=288 * 5, monthly=288 * 22) -> pd.DataFrame:
    """HAR-RV components (Corsi 2009): daily / weekly-avg / monthly-avg realized vol —
    same feature family as the daily/weekly/monthly RV inputs in the existing production
    HAR-RV+LightGBM pipeline, for consistency.
    """
    rv_daily = log_ret.rolling(daily, min_periods=_min_periods(daily)).std()
    rv_weekly = log_ret.rolling(weekly, min_periods=_min_periods(weekly)).std()
    rv_monthly = log_ret.rolling(monthly, min_periods=_min_periods(monthly)).std()
    return pd.DataFrame({
        "har_rv_daily": rv_daily,
        "har_rv_weekly": rv_weekly,
        "har_rv_monthly": rv_monthly,
    }, index=log_ret.index)


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                  window: int) -> pd.Series:
    gk_term = 0.5 * np.log(high / low) ** 2 - (2 * _LN2 - 1) * np.log(close / open_) ** 2
    var = gk_term.rolling(window, min_periods=_min_periods(window)).mean().clip(lower=0)
    return np.sqrt(var)


def yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
               window: int) -> pd.Series:
    """Handles jumps between consecutive bars (the M5 analogue of an overnight gap)
    better than Garman-Klass, at the cost of needing the previous bar's close.
    """
    prev_close = close.shift(1)
    co = np.log(open_ / prev_close)
    oc = np.log(close / open_)
    rs_term = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)

    mp = _min_periods(window)
    sigma_o2 = co.rolling(window, min_periods=mp).var()
    sigma_c2 = oc.rolling(window, min_periods=mp).var()
    sigma_rs2 = rs_term.rolling(window, min_periods=mp).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = (sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2).clip(lower=0)
    return np.sqrt(yz_var)
