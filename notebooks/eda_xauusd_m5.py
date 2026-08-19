"""EDA on XAUUSD M5 (2005-2026) — exploratory only, not production code.

Answers the questions that actually drive downstream design decisions:
  - is the price series non-stationary / are returns stationary (motivates FFD)?
  - is there volatility clustering (motivates GARCH)?
  - how fat are the tails, and how does that interact with the p90/p95 TP quantiles?
  - do gaps/regimes/seasonality require session-aware features or fold boundaries?
  - what does the actual N=12-candle forward-return distribution look like (the label)?

Run: ./.venv/bin/python3 notebooks/eda_xauusd_m5.py
Outputs: notebooks/figures/*.png + notebooks/eda_summary.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import fftconvolve
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SUMMARY = {}


def load_symbol(symbol: str, interval: str) -> pd.DataFrame:
    files = sorted((DATA_RAW / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


print("Loading XAUUSD M5...")
xau = load_symbol("XAUUSD", "M5")
xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2
xau["mid_open"] = (xau["bid_open"] + xau["ask_open"]) / 2
xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
xau["spread"] = xau["ask_close"] - xau["bid_close"]

# --- gap-aware returns: a diff() across a data gap is NOT a 5-minute return and must
# not be pooled with real 5-min returns (it silently produced a fake volatility spike
# in early testing — see data-quality section below). Both weekend/holiday closures and
# genuine intraday data holes (worse in 2005-2012, the thin end of the free feed) are
# excluded from the "clean" return series used for all distributional/stats work.
xau["time_diff_min"] = xau["time"].diff().dt.total_seconds() / 60
is_clean_step = xau["time_diff_min"] == 5
is_weekend_gap = xau["time_diff_min"] > 1000
is_anomalous_gap = (xau["time_diff_min"] > 5) & (xau["time_diff_min"] <= 1000)

xau["raw_log_ret"] = np.log(xau["mid_close"]).diff()
xau["log_ret"] = xau["raw_log_ret"].where(is_clean_step)

SUMMARY["n_bars"] = int(len(xau))
SUMMARY["date_range"] = [str(xau["time"].min()), str(xau["time"].max())]
print(f"  {len(xau):,} bars, {xau['time'].min()} -> {xau['time'].max()}")

print("0. Data quality: gap audit...")
xau["year"] = xau["time"].dt.year
gaps_by_year = xau.loc[is_anomalous_gap].groupby("year").size()
weekend_return = xau.loc[is_weekend_gap, "raw_log_ret"].dropna()
SUMMARY["data_quality"] = {
    "clean_5min_steps": int(is_clean_step.sum()),
    "clean_5min_steps_pct": float(is_clean_step.mean() * 100),
    "weekend_holiday_gaps": int(is_weekend_gap.sum()),
    "anomalous_intraday_gaps": int(is_anomalous_gap.sum()),
    "anomalous_gaps_by_year": {str(k): int(v) for k, v in gaps_by_year.items()},
    "weekend_gap_return_std": float(weekend_return.std()),
    "weekend_gap_return_p05_p95": [float(weekend_return.quantile(0.05)), float(weekend_return.quantile(0.95))],
    "note": ("2005-2012 has 3-10x more intraday data gaps than 2013+ (thin early tick "
             "coverage on the free feed) — all gap-spanning bars are excluded from return "
             "stats below, not just flagged."),
}
print(f"  clean steps: {is_clean_step.mean()*100:.2f}%, anomalous intraday gaps: {is_anomalous_gap.sum()}"
      f" (concentrated pre-2013: {gaps_by_year.reindex(range(2005,2013), fill_value=0).sum()} of {is_anomalous_gap.sum()})")

xau_clean = xau.dropna(subset=["log_ret"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1. Return distribution / fat tails
# ---------------------------------------------------------------------------
print("1. Return distribution...")
ret = xau_clean["log_ret"].values
SUMMARY["returns"] = {
    "mean": float(np.mean(ret)),
    "std": float(np.std(ret)),
    "skew": float(stats.skew(ret)),
    "kurtosis_excess": float(stats.kurtosis(ret)),  # 0 for normal
    "jarque_bera_stat": float(stats.jarque_bera(ret)[0]),
    "jarque_bera_pvalue": float(stats.jarque_bera(ret)[1]),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(ret, bins=300, density=True, alpha=0.7, color="#2b6cb0", label="XAUUSD M5 log-returns")
xs = np.linspace(ret.min(), ret.max(), 500)
axes[0].plot(xs, stats.norm.pdf(xs, np.mean(ret), np.std(ret)), color="#e53e3e", lw=1.5, label="Normal fit")
axes[0].set_yscale("log")
axes[0].set_title("Return distribution vs Normal (log scale)")
axes[0].legend()
stats.probplot(ret, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs Normal")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_return_distribution.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 2. Stationarity: price level vs returns, + fractional differentiation sweep
# ---------------------------------------------------------------------------
print("2. Stationarity / fractional differentiation...")


def get_weights_ffd(d, thres=1e-5, max_size=500):
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: np.ndarray, d: float, thres=1e-5):
    w = get_weights_ffd(d, thres)
    if len(w) >= len(series):
        return np.array([]), w
    out = fftconvolve(series, w, mode="valid")
    return out, w


adf_price = adfuller(xau["mid_close"].values, maxlag=20, autolag=None)
adf_ret = adfuller(ret, maxlag=20, autolag=None)
SUMMARY["adf"] = {
    "price_level_pvalue": float(adf_price[1]),
    "returns_pvalue": float(adf_ret[1]),
}
print(f"  ADF price level p={adf_price[1]:.4f} (expect non-stationary), returns p={adf_ret[1]:.4f} (expect stationary)")

price = xau["mid_close"].values
ffd_results = []
for d in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
    diffed, w = frac_diff_ffd(price, d)
    if len(diffed) < 1000:
        continue
    p_adf = adfuller(diffed, maxlag=20, autolag=None)[1]
    corr_with_price = float(np.corrcoef(diffed, price[-len(diffed):])[0, 1])
    ffd_results.append({"d": d, "window": len(w), "adf_pvalue": p_adf, "corr_with_price": corr_with_price})
    print(f"  d={d}: window={len(w)}, ADF p={p_adf:.4f}, corr(diffed, price)={corr_with_price:.3f}")

SUMMARY["ffd_sweep"] = ffd_results
min_d_stationary = next((r["d"] for r in ffd_results if r["adf_pvalue"] < 0.05), None)
SUMMARY["ffd_min_d_for_stationarity"] = min_d_stationary

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ds = [r["d"] for r in ffd_results]
ax1.plot(ds, [r["adf_pvalue"] for r in ffd_results], "o-", color="#e53e3e", label="ADF p-value")
ax1.axhline(0.05, color="gray", ls="--", lw=1)
ax1.set_xlabel("d")
ax1.set_ylabel("ADF p-value", color="#e53e3e")
ax2 = ax1.twinx()
ax2.plot(ds, [r["corr_with_price"] for r in ffd_results], "s-", color="#2b6cb0", label="corr with price")
ax2.set_ylabel("Correlation with original price (memory retained)", color="#2b6cb0")
plt.title("FFD: stationarity vs memory retention trade-off")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_ffd_sweep.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 3. Volatility clustering (ARCH effects) — motivates GARCH
# ---------------------------------------------------------------------------
print("3. Volatility clustering...")
abs_ret = np.abs(ret)
sq_ret = ret ** 2
acf_ret = acf(ret, nlags=50, fft=True)
acf_absret = acf(abs_ret, nlags=50, fft=True)
lb_sq = acorr_ljungbox(sq_ret, lags=[10, 20, 50], return_df=True)
SUMMARY["volatility_clustering"] = {
    "ljung_box_squared_returns_pvalues": {str(k): float(v) for k, v in
                                           zip(lb_sq.index, lb_sq["lb_pvalue"].values)},
    "acf_returns_lag1": float(acf_ret[1]),
    "acf_abs_returns_lag1": float(acf_absret[1]),
}
print(f"  ACF(returns, lag1)={acf_ret[1]:.4f} (near 0 expected), ACF(|returns|, lag1)={acf_absret[1]:.4f} (clustering if >>0)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].stem(range(51), acf_ret, markerfmt=" ", basefmt=" ")
axes[0].set_title("ACF of raw returns (weak-form efficiency check)")
axes[1].stem(range(51), acf_absret, markerfmt=" ", basefmt=" ")
axes[1].set_title("ACF of |returns| (volatility clustering / ARCH effect)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_acf_returns_vs_absreturns.png", dpi=110)
plt.close()

# how many bars until autocorrelation of |returns| decays near zero (informs embargo width)
threshold = 0.05
decay_lag = next((i for i in range(1, 51) if acf_absret[i] < threshold), 50)
SUMMARY["volatility_clustering"]["decay_below_0.05_at_lag"] = int(decay_lag)
print(f"  |returns| autocorrelation decays below 0.05 at lag {decay_lag} (~{decay_lag*5} minutes)")

# ---------------------------------------------------------------------------
# 4. Seasonality: intraday and day-of-week volatility
# ---------------------------------------------------------------------------
print("4. Seasonality...")
xau_clean["hour"] = xau_clean["time"].dt.hour
xau_clean["dow"] = xau_clean["time"].dt.dayofweek
hourly_vol = xau_clean.groupby("hour")["log_ret"].apply(lambda x: x.std())
dow_vol = xau_clean.groupby("dow")["log_ret"].apply(lambda x: x.std())
SUMMARY["seasonality"] = {
    "hourly_vol_std": {str(k): float(v) for k, v in hourly_vol.items()},
    "dow_vol_std": {str(k): float(v) for k, v in dow_vol.items()},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
hourly_vol.plot(kind="bar", ax=axes[0], color="#2b6cb0")
axes[0].set_title("Return std by hour of day (UTC)")
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_vol.index = [dow_names[i] for i in dow_vol.index]
dow_vol.plot(kind="bar", ax=axes[1], color="#2b6cb0")
axes[1].set_title("Return std by day of week")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_seasonality.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 5. Regime evolution: rolling realized volatility over 21 years
# ---------------------------------------------------------------------------
print("5. Regime evolution...")
xau_daily = xau_clean.set_index("time")["log_ret"].resample("1D").std().dropna()
rolling_vol_30d = xau_daily.rolling(30).mean()

fig, ax = plt.subplots(figsize=(12, 4.5))
rolling_vol_30d.plot(ax=ax, color="#2b6cb0")
ax.set_title("30-day rolling average of daily M5-return std (regime evolution, 2005-2026)")
ax.set_ylabel("avg daily M5 return std")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_regime_evolution.png", dpi=110)
plt.close()

SUMMARY["regime"] = {
    "vol_percentile_10": float(rolling_vol_30d.quantile(0.10)),
    "vol_percentile_50": float(rolling_vol_30d.quantile(0.50)),
    "vol_percentile_90": float(rolling_vol_30d.quantile(0.90)),
    "vol_max": float(rolling_vol_30d.max()),
    "vol_max_date": str(rolling_vol_30d.idxmax()),
}
print(f"  peak volatility regime: {SUMMARY['regime']['vol_max_date']}")

# ---------------------------------------------------------------------------
# 6. Spread regime: cost of execution over time and vs volatility
# ---------------------------------------------------------------------------
print("6. Spread regime...")
xau_clean["abs_ret"] = abs_ret
spread_vol_corr = float(xau_clean["spread"].corr(xau_clean["abs_ret"]))
SUMMARY["spread"] = {
    "mean": float(xau_clean["spread"].mean()),
    "median": float(xau_clean["spread"].median()),
    "p95": float(xau_clean["spread"].quantile(0.95)),
    "p99": float(xau_clean["spread"].quantile(0.99)),
    "corr_with_abs_return": spread_vol_corr,
}
print(f"  spread mean={SUMMARY['spread']['mean']:.4f}, corr(spread, |return|)={spread_vol_corr:.3f}")

spread_daily = xau_clean.set_index("time")["spread"].resample("1D").mean().dropna()
fig, ax = plt.subplots(figsize=(12, 4.5))
spread_daily.rolling(30).mean().plot(ax=ax, color="#dd6b20")
ax.set_title("30-day rolling average spread (execution cost regime, 2005-2026)")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_spread_regime.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 7. Cross-asset: gold vs DXY, gold vs USTBOND (rate proxy)
# ---------------------------------------------------------------------------
print("7. Cross-asset correlation...")
dxy = load_symbol("DXY", "M5")
dxy["mid_close"] = dxy["close"] if "close" in dxy.columns else dxy["bid_close"] if "bid_close" in dxy.columns else None
if dxy["mid_close"] is None:
    dxy["mid_close"] = dxy["close"]
dxy["log_ret_dxy"] = np.log(dxy["mid_close"]).diff()

ustbond = load_symbol("USTBOND", "M5")
if "close" in ustbond.columns:
    ustbond["mid_close"] = ustbond["close"]
ustbond["log_ret_ust"] = np.log(ustbond["mid_close"]).diff()

merged_dxy = pd.merge_asof(
    xau_clean[["time", "log_ret"]].sort_values("time"),
    dxy[["time", "log_ret_dxy"]].dropna().sort_values("time"),
    on="time", tolerance=pd.Timedelta("5min"), direction="nearest",
).dropna()
merged_ust = pd.merge_asof(
    xau_clean[["time", "log_ret"]].sort_values("time"),
    ustbond[["time", "log_ret_ust"]].dropna().sort_values("time"),
    on="time", tolerance=pd.Timedelta("5min"), direction="nearest",
).dropna()

corr_dxy = float(merged_dxy["log_ret"].corr(merged_dxy["log_ret_dxy"]))
corr_ust = float(merged_ust["log_ret"].corr(merged_ust["log_ret_ust"]))
SUMMARY["cross_asset"] = {
    "corr_gold_dxy_full_period": corr_dxy,
    "corr_gold_ustbond_full_period": corr_ust,
    "dxy_n_obs": int(len(merged_dxy)),
    "ustbond_n_obs": int(len(merged_ust)),
}
print(f"  corr(gold_ret, DXY_ret)={corr_dxy:.3f} (expect negative), corr(gold_ret, USTBOND_ret)={corr_ust:.3f}")

# rolling 60-day correlation gold vs DXY (relationship stability check)
merged_dxy_daily = merged_dxy.set_index("time")[["log_ret", "log_ret_dxy"]].resample("1D").sum()
rolling_corr = merged_dxy_daily["log_ret"].rolling(60).corr(merged_dxy_daily["log_ret_dxy"])
fig, ax = plt.subplots(figsize=(12, 4.5))
rolling_corr.plot(ax=ax, color="#805ad5")
ax.axhline(0, color="gray", lw=1)
ax.set_title("60-day rolling correlation: gold returns vs DXY returns")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_rolling_corr_gold_dxy.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# 8. Target distribution: forward log-return at N=12 candles (the actual label)
# ---------------------------------------------------------------------------
print("8. Forward-return target distribution (N=12 candles, ~1h)...")
N = 12
# valid only if the N-candle window doesn't span a data/weekend gap — otherwise "12
# candles forward" silently becomes "whatever time a gap happens to cover"
fwd_time_ok = (xau["time"].shift(-N) - xau["time"]) == pd.Timedelta(minutes=5 * N)
xau["fwd_ret_12"] = np.log(xau["mid_close"].shift(-N) / xau["mid_close"]).where(fwd_time_ok)
fwd = xau["fwd_ret_12"].dropna()
print(f"  valid {N}-candle windows: {len(fwd):,} of {len(xau):,} ({len(fwd)/len(xau)*100:.2f}%)")

quantile_levels = [0.05, 0.10, 0.50, 0.90, 0.95]
fwd_quantiles = {str(q): float(fwd.quantile(q)) for q in quantile_levels}
SUMMARY["forward_return_target"] = {
    "horizon_candles": N,
    "unconditional_quantiles": fwd_quantiles,
    "std": float(fwd.std()),
    "skew": float(stats.skew(fwd)),
}
print("  unconditional quantiles:", fwd_quantiles)

# conditional on volatility regime tercile (does the tail move a lot by regime?)
xau["vol_20"] = xau["log_ret"].rolling(20).std()
xau_valid = xau.dropna(subset=["vol_20", "fwd_ret_12"]).copy()
xau_valid["vol_tercile"] = pd.qcut(xau_valid["vol_20"], 3, labels=["low", "mid", "high"])
cond_quantiles = {}
for tercile in ["low", "mid", "high"]:
    sub = xau_valid.loc[xau_valid["vol_tercile"] == tercile, "fwd_ret_12"]
    cond_quantiles[tercile] = {str(q): float(sub.quantile(q)) for q in quantile_levels}
SUMMARY["forward_return_target"]["quantiles_by_vol_tercile"] = cond_quantiles
print("  quantiles by volatility tercile:", json.dumps(cond_quantiles, indent=2))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(fwd, bins=400, density=True, alpha=0.7, color="#2b6cb0")
for q, v in fwd_quantiles.items():
    ax.axvline(v, color="#e53e3e", lw=1, ls="--")
    ax.text(v, ax.get_ylim()[1] * 0.9, f"p{float(q)*100:.0f}", rotation=90, fontsize=8)
ax.set_xlim(fwd.quantile(0.001), fwd.quantile(0.999))
ax.set_title(f"Forward {N}-candle log-return distribution (the quantile-regression target)")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_forward_return_target.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
for tercile, color in zip(["low", "mid", "high"], ["#38a169", "#dd6b20", "#e53e3e"]):
    sub = xau_valid.loc[xau_valid["vol_tercile"] == tercile, "fwd_ret_12"]
    ax.hist(sub, bins=200, density=True, alpha=0.4, label=f"{tercile} vol", color=color)
ax.set_xlim(fwd.quantile(0.001), fwd.quantile(0.999))
ax.legend()
ax.set_title("Forward-return distribution by current-volatility tercile")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_forward_return_by_vol_tercile.png", dpi=110)
plt.close()

# ---------------------------------------------------------------------------
with open(Path(__file__).resolve().parent / "eda_summary.json", "w") as f:
    json.dump(SUMMARY, f, indent=2)

print("\nDone. Figures in notebooks/figures/, summary in notebooks/eda_summary.json")
