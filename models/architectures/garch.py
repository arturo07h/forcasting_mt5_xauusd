"""GARCH(1,1) with skew-t innovations — the explicit time-series baseline (as opposed to
the tabular ML/DL candidates). Fit once per fold on daily-aggregated returns up to
train_end (GARCH's MLE doesn't fit well or fast on 1M+ raw 5-minute points, and fitting
volatility models on daily data is standard practice); the conditional variance is then
propagated through the test period's *realized* daily returns using the fixed fitted
(omega, alpha, beta) — no re-estimation on test data, so this is a legitimate walk-forward
forecast, not leakage.

The daily forecast is scaled down to the N=12-candle (~1h) horizon by the bar-count
fraction of a trading day (12/288) — this assumes intraday volatility density is roughly
uniform, a known simplification flagged here rather than hidden. It is a materially
cruder approximation than the tabular models' bar-by-bar conditional estimate, which is
exactly why it's the baseline being compared against, not assumed to win.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent

BARS_PER_DAY = 288
HORIZON_BARS = 12
HORIZON_FRACTION = HORIZON_BARS / BARS_PER_DAY


def fit_and_forecast(xau_m5: pd.DataFrame, train_end: pd.Timestamp, test_end: pd.Timestamp,
                      quantiles: list[float]) -> pd.DataFrame:
    """Returns a DataFrame indexed by day (available_time = day+1, i.e. when that day's
    return becomes fully known) with columns pred_q{q} for each quantile — to be as-of
    joined onto M5 bars the same way higher_timeframe.py joins H1/H4 context.
    """
    daily = xau_m5.set_index("time")["mid_close"].resample("1D").last().dropna()
    daily_ret_pct = 100 * np.log(daily).diff().dropna()

    train_ret = daily_ret_pct[daily_ret_pct.index < train_end]
    am = arch_model(train_ret, mean="Constant", vol="GARCH", p=1, q=1, dist="skewt")
    res = am.fit(disp="off")
    omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    mu, eta, lam = res.params["mu"], res.params["eta"], res.params["lambda"]

    test_ret = daily_ret_pct[(daily_ret_pct.index >= train_end) & (daily_ret_pct.index < test_end)]
    if len(test_ret) == 0:
        return pd.DataFrame()

    sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
    last_resid2 = float((train_ret.iloc[-1] - mu) ** 2)

    sigma2_path = []
    for r in test_ret.values:
        sigma2 = omega + alpha * last_resid2 + beta * sigma2
        sigma2_path.append(sigma2)
        last_resid2 = (r - mu) ** 2
    sigma2_path = np.array(sigma2_path)

    sigma_h_pct = np.sqrt(sigma2_path * HORIZON_FRACTION)  # still in "percent return" units
    mu_h_pct = mu * HORIZON_FRACTION

    skewt = SkewStudent()
    out = {"time": test_ret.index + pd.Timedelta(days=1)}  # available once that day's return is known
    for q in quantiles:
        z_q = float(skewt.ppf(q, [eta, lam]))
        out[f"pred_q{q}"] = (mu_h_pct + z_q * sigma_h_pct) / 100  # back to raw log-return units
    return pd.DataFrame(out)


def merge_onto_m5(m5_time: pd.Series, daily_forecast: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    pred_cols = [f"pred_q{q}" for q in quantiles]
    left = pd.DataFrame({"time": m5_time}).sort_values("time")
    left["time"] = left["time"].astype("datetime64[us, UTC]")
    right = daily_forecast.sort_values("time").copy()
    right["time"] = right["time"].astype("datetime64[us, UTC]")
    merged = pd.merge_asof(left, right, on="time", direction="backward")
    return merged[pred_cols].set_index(left.index)
