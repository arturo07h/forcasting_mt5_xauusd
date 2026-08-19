"""Fixed-width-window fractional differentiation (López de Prado). EDA found d≈0.35-0.4
is the minimal order that passes ADF on this series — see notebooks/eda_xauusd_m5.py.
d should still be re-validated (not blindly reused) inside each walk-forward training
fold, since the minimal-d point can drift as more data enters the fold.
"""
import numpy as np
from scipy.signal import fftconvolve
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, thres: float = 1e-5, max_size: int = 500) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: np.ndarray, d: float, thres: float = 1e-5) -> np.ndarray:
    """Returns an array shorter than `series` by (window-1); left-pad with NaN by the
    caller if you need to align it back to the original index.
    """
    w = get_weights_ffd(d, thres)
    if len(w) >= len(series):
        raise ValueError(f"window ({len(w)}) >= series length ({len(series)}) for d={d}")
    return fftconvolve(series, w, mode="valid")


def find_min_stationary_d(series: np.ndarray, d_grid=(0.2, 0.3, 0.35, 0.4, 0.5, 0.6),
                           thres: float = 1e-5, p_value: float = 0.05) -> float | None:
    """Smallest d in d_grid whose FFD series passes ADF at the given p-value."""
    for d in d_grid:
        diffed = frac_diff_ffd(series, d, thres)
        if adfuller(diffed, maxlag=20, autolag=None)[1] < p_value:
            return d
    return None
