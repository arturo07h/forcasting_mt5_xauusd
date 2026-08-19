"""Triple-barrier labeling (López de Prado): scans the actual forward high/low path to
find which barrier — upper (TP), lower (SL), or vertical (timeout) — is touched first,
rather than only looking at the endpoint return like the EDA's forward-return target did.

Barrier *distances* are a parameter, not fixed here — this module doesn't know or care
whether they come from a placeholder rule or a trained model. See label_dataset() below
for the placeholder rule used until the quantile-regression model exists: SL from ATR,
TP from the EDA-derived p95 magnitude scaled by the current/average volatility regime.
Once that model exists, its predicted quantiles replace this rule directly — the core
scan (triple_barrier_scan) doesn't change.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _scan(high, low, close, upper_dist, lower_dist, max_horizon, valid):
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    ret = np.full(n, np.nan)
    time_to_hit = np.full(n, -1, dtype=np.int32)

    for t in range(n):
        if not valid[t]:
            continue
        entry = close[t]
        upper = entry + upper_dist[t]
        lower = entry - lower_dist[t]
        horizon_end = min(t + max_horizon, n - 1)

        hit = 0
        hit_ret = np.nan
        hit_time = horizon_end - t
        for j in range(t + 1, horizon_end + 1):
            up_hit = high[j] >= upper
            down_hit = low[j] <= lower
            if up_hit and down_hit:
                # both touched in the same bar — can't tell which came first from
                # OHLC alone; conservative assumption is the adverse outcome (SL)
                hit = -1
                hit_ret = (lower - entry) / entry
                hit_time = j - t
                break
            if up_hit:
                hit = 1
                hit_ret = (upper - entry) / entry
                hit_time = j - t
                break
            if down_hit:
                hit = -1
                hit_ret = (lower - entry) / entry
                hit_time = j - t
                break
        if hit == 0:
            hit_ret = (close[horizon_end] - entry) / entry

        label[t] = hit
        ret[t] = hit_ret
        time_to_hit[t] = hit_time

    return label, ret, time_to_hit


def triple_barrier_scan(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         upper_dist: np.ndarray, lower_dist: np.ndarray,
                         max_horizon: int, valid: np.ndarray):
    """upper_dist / lower_dist are absolute price distances from entry (same array
    length as close). valid[t]=False skips labeling that bar entirely (label=0,
    ret=NaN) — use this to exclude windows that would scan across a bad data gap.

    Returns (label, ret, time_to_hit):
      label: +1 TP hit, -1 SL hit, 0 timeout/invalid
      ret:   realized return at whichever barrier/timeout ended the trade
      time_to_hit: bars until exit (horizon_end-t if timeout)
    """
    return _scan(high.astype(np.float64), low.astype(np.float64), close.astype(np.float64),
                 upper_dist.astype(np.float64), lower_dist.astype(np.float64),
                 int(max_horizon), valid.astype(np.bool_))
