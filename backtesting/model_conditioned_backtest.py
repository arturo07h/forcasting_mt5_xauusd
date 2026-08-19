"""The real go/no-go test: does the trained model's *actual* predicted p90/p95 (vs the
placeholder EDA-average-scaled rule in labeling/build_labels.py) produce a meaningfully
better R-multiple / geometric-growth profile at 5% fixed risk than the unconditional
blind-entry baseline in backtesting/baseline_risk_check.py?

Uses LightGBM's out-of-sample test-fold predictions (the winning model on pinball loss)
— these are genuine walk-forward OOS predictions, never seen during that model's
training. SL stays ATR-based (unchanged, consistent with the project's SL-by-volatility
decision); TP is now the model's own predicted quantile at each bar instead of a flat
vol-scaled placeholder.

Run: ./.venv/bin/python3 backtesting/model_conditioned_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
from labeling.triple_barrier import triple_barrier_scan
from backtesting.baseline_risk_check import sequential_r_multiples, max_streak

MAX_HORIZON = 48
SL_ATR_MULT = 1.5
TP_QUANTILE = "pred_q0.9"  # switch to pred_q0.95 to compare — both tested below
RISK_FRACTIONS_TO_TEST = [0.05, 0.03, 0.02, 0.01, 0.005]


def load_symbol(symbol, interval):
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def run(tp_quantile_col: str = TP_QUANTILE):
    print(f"Using {tp_quantile_col} as the model-predicted TP quantile...")
    preds = pd.read_parquet(PROJECT_ROOT / "models" / "checkpoints" / "lightgbm_core_predictions.parquet")
    preds["time"] = pd.to_datetime(preds["time"], utc=True)

    xau = load_symbol("XAUUSD", "M5")
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    feat_files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5").glob("year=*/month=*.parquet"))
    feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)[["time", "atr_14"]]
    feat["time"] = pd.to_datetime(feat["time"])

    df = xau.merge(feat, on="time", how="inner").merge(preds[["time", tp_quantile_col]], on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  {len(df):,} OOS test bars with predictions + ATR")

    time_diff_min = df["time"].diff().dt.total_seconds() / 60
    is_anomalous_gap = (time_diff_min > 5) & (time_diff_min <= 1000)
    gap_shifted = is_anomalous_gap.shift(-1)
    bad_gap_ahead = gap_shifted[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].fillna(0).astype(bool)
    not_near_end = np.arange(len(df)) < (len(df) - MAX_HORIZON - 1)
    valid = (~bad_gap_ahead) & not_near_end & df["atr_14"].notna() & (df[tp_quantile_col] > 0)
    print(f"  {valid.sum():,} valid starting bars ({valid.mean()*100:.1f}%) — "
          f"model-predicted TP <= 0 bars excluded (no long-side signal)")

    sl_dist = SL_ATR_MULT * df["atr_14"]
    tp_dist = df["mid_close"] * df[tp_quantile_col]

    label, ret, time_to_hit = triple_barrier_scan(
        df["mid_high"].values, df["mid_low"].values, df["mid_close"].values,
        tp_dist.values, sl_dist.values, MAX_HORIZON, valid.values,
    )
    df["label"], df["realized_ret"], df["time_to_hit"] = label, ret, time_to_hit
    df["sl_dist"], df["tp_dist"] = sl_dist, tp_dist
    labeled = df.loc[valid].reset_index(drop=True)

    print(f"\n{len(labeled):,} model-conditioned labeled bars")
    print(labeled["label"].value_counts(normalize=True).rename({1: "TP", -1: "SL", 0: "timeout"}))

    seq_R, seq_label = sequential_r_multiples(labeled)
    n_trades = len(seq_R)
    streak = max_streak(seq_label == -1)
    print(f"\n{n_trades:,} sequential trades")
    print(f"SL: {(seq_label==-1).mean()*100:.1f}%  TP: {(seq_label==1).mean()*100:.1f}%  "
          f"timeout: {(seq_label==0).mean()*100:.1f}%")
    print(f"Arithmetic E[R]: {seq_R.mean():.4f}")
    print(f"Max consecutive SL streak: {streak}")

    for f in RISK_FRACTIONS_TO_TEST:
        growth = np.mean(np.log(1 + f * seq_R))
        verdict = "RUIN" if growth < 0 else "survives"
        print(f"  risk={f*100:.1f}%: growth/trade={growth:.6f} -> {verdict}")

    fs = np.linspace(0.001, 0.10, 200)
    growths = [np.mean(np.log(1 + f * seq_R)) for f in fs]
    best_f = float(fs[int(np.argmax(growths))])
    print(f"\nGrowth-optimal fixed fraction: {best_f*100:.2f}%")
    return seq_R, seq_label


def run_conviction_sweep(tp_quantile_col: str = TP_QUANTILE, percentiles=(0, 50, 75, 90)):
    """Does filtering for the model's highest-conviction signals (largest predicted TP
    relative to ATR) close the gap to 5% being survivable? Selectivity is the intended
    mechanism (the model's whole job vs. the unconditional/blind-entry baseline), so
    this checks how far it actually gets, not just whether it helps in principle.
    """
    preds = pd.read_parquet(PROJECT_ROOT / "models" / "checkpoints" / "lightgbm_core_predictions.parquet")
    preds["time"] = pd.to_datetime(preds["time"], utc=True)

    xau = load_symbol("XAUUSD", "M5")
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    feat_files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5").glob("year=*/month=*.parquet"))
    feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)[["time", "atr_14"]]
    feat["time"] = pd.to_datetime(feat["time"])

    df = xau.merge(feat, on="time", how="inner").merge(preds[["time", tp_quantile_col]], on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)

    time_diff_min = df["time"].diff().dt.total_seconds() / 60
    is_anomalous_gap = (time_diff_min > 5) & (time_diff_min <= 1000)
    gap_shifted = is_anomalous_gap.shift(-1)
    bad_gap_ahead = gap_shifted[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].fillna(0).astype(bool)
    not_near_end = np.arange(len(df)) < (len(df) - MAX_HORIZON - 1)

    conviction = df[tp_quantile_col] / df["atr_14"].clip(lower=1e-6) * df["mid_close"]
    sl_dist = SL_ATR_MULT * df["atr_14"]
    tp_dist = df["mid_close"] * df[tp_quantile_col]

    results = []
    for pct in percentiles:
        thresh = np.nanpercentile(conviction, pct) if pct > 0 else -np.inf
        valid = ((~bad_gap_ahead) & not_near_end & df["atr_14"].notna()
                  & (df[tp_quantile_col] > 0) & (conviction >= thresh))

        label, ret, time_to_hit = triple_barrier_scan(
            df["mid_high"].values, df["mid_low"].values, df["mid_close"].values,
            tp_dist.values, sl_dist.values, MAX_HORIZON, valid.values,
        )
        d2 = df.copy()
        d2["label"], d2["realized_ret"], d2["time_to_hit"] = label, ret, time_to_hit
        d2["sl_dist"], d2["tp_dist"] = sl_dist, tp_dist
        labeled = d2.loc[valid].reset_index(drop=True)

        seq_R, seq_label = sequential_r_multiples(labeled)
        fs = np.linspace(0.001, 0.10, 200)
        growths = [np.mean(np.log(1 + f * seq_R)) for f in fs]
        best_f = float(fs[int(np.argmax(growths))])
        growth_at_5pct = float(np.mean(np.log(1 + 0.05 * seq_R)))

        row = {
            "top_pct_by_conviction": 100 - pct, "n_trades": int(len(seq_R)),
            "sl_rate": float((seq_label == -1).mean()), "tp_rate": float((seq_label == 1).mean()),
            "arithmetic_E_R": float(seq_R.mean()), "growth_optimal_fraction": best_f,
            "growth_at_5pct_risk": growth_at_5pct,
        }
        results.append(row)
        print(f"top {row['top_pct_by_conviction']}% conviction: n={row['n_trades']:,} "
              f"SL={row['sl_rate']*100:.1f}% TP={row['tp_rate']*100:.1f}% E[R]={row['arithmetic_E_R']:.4f} "
              f"growth-optimal-f={best_f*100:.2f}% growth@5%={growth_at_5pct:.6f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    import json

    print("=" * 60, "\nTP = predicted p90\n", "=" * 60)
    run("pred_q0.9")
    print("\n" + "=" * 60, "\nTP = predicted p95\n", "=" * 60)
    run("pred_q0.95")

    print("\n" + "=" * 60, "\nConviction (selectivity) sweep, TP = predicted p90\n", "=" * 60)
    sweep = run_conviction_sweep("pred_q0.9", percentiles=(0, 50, 75, 90, 95))
    out_path = Path(__file__).resolve().parent / "model_conditioned_backtest_summary.json"
    out_path.write_text(sweep.to_json(orient="records", indent=2))
    print(f"\nSaved: {out_path}")
