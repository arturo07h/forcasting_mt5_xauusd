"""Session/seasonality features. EDA showed a clear intraday vol pattern peaking at
12-15 UTC (London/NY overlap) — cyclical encoding avoids the hour=23/hour=0 discontinuity
a raw integer would create.
"""
import numpy as np
import pandas as pd


def session_features(time: pd.Series) -> pd.DataFrame:
    hour = time.dt.hour + time.dt.minute / 60
    dow = time.dt.dayofweek

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "session_asia": ((hour >= 0) & (hour < 8)).astype(int),
        "session_london": ((hour >= 8) & (hour < 16)).astype(int),
        "session_ny": ((hour >= 13) & (hour < 21)).astype(int),
        "session_london_ny_overlap": ((hour >= 13) & (hour < 16)).astype(int),
    }, index=time.index)
