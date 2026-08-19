"""The meta-labeling-filtered backtest — does a proper "should I take this trade"
classifier (not just a magnitude-based conviction proxy) close the gap to 5% fixed
risk being survivable? Answer: yes, with real but bounded confidence, and a severe
drawdown path even when it works. See the printed caveats — this is not a clean "problem
solved," it's a materially better and more honestly uncertain result than the earlier
conviction-filter sweep.

Run: ./.venv/bin/python3 backtesting/meta_labeled_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT
from labeling.triple_barrier import triple_barrier_scan
from backtesting.baseline_risk_check import sequential_r_multiples, max_streak
from validation.walk_forward import FOLDS

MAX_HORIZON = 48
SL_ATR_MULT = 1.5
TP_QUANTILE_COL = "pred_q0.9"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
N_BOOTSTRAP = 2000


def load_symbol(symbol, interval):
    files = sorted((DATA_RAW_DIR / f"symbol={symbol}" / f"interval={interval}").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def build_dataset():
    preds = pd.read_parquet(PROJECT_ROOT / "models" / "checkpoints" / "lightgbm_core_predictions.parquet")
    preds["time"] = pd.to_datetime(preds["time"], utc=True)
    meta = pd.read_parquet(PROJECT_ROOT / "models" / "checkpoints" / "meta_label_core_predictions.parquet")
    meta["time"] = pd.to_datetime(meta["time"], utc=True)

    xau = load_symbol("XAUUSD", "M5")
    xau["mid_high"] = (xau["bid_high"] + xau["ask_high"]) / 2
    xau["mid_low"] = (xau["bid_low"] + xau["ask_low"]) / 2
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    feat_files = sorted((DATA_PROCESSED_DIR / "symbol=XAUUSD" / "interval=M5").glob("year=*/month=*.parquet"))
    feat = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)[["time", "atr_14"]]
    feat["time"] = pd.to_datetime(feat["time"])

    df = (xau.merge(feat, on="time", how="inner")
             .merge(preds[["time", TP_QUANTILE_COL, "fold"]], on="time", how="inner")
             .merge(meta[["time", "meta_proba"]], on="time", how="inner"))
    return df.sort_values("time").reset_index(drop=True)


def label_at_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    time_diff_min = df["time"].diff().dt.total_seconds() / 60
    is_anomalous_gap = (time_diff_min > 5) & (time_diff_min <= 1000)
    gap_shifted = is_anomalous_gap.shift(-1)
    bad_gap_ahead = gap_shifted[::-1].rolling(MAX_HORIZON, min_periods=1).max()[::-1].fillna(0).astype(bool)
    not_near_end = np.arange(len(df)) < (len(df) - MAX_HORIZON - 1)

    valid = ((~bad_gap_ahead) & not_near_end & df["atr_14"].notna()
              & (df[TP_QUANTILE_COL] > 0) & (df["meta_proba"] >= threshold))

    sl_dist = SL_ATR_MULT * df["atr_14"]
    tp_dist = df["mid_close"] * df[TP_QUANTILE_COL]
    label, ret, time_to_hit = triple_barrier_scan(
        df["mid_high"].values, df["mid_low"].values, df["mid_close"].values,
        tp_dist.values, sl_dist.values, MAX_HORIZON, valid.values,
    )
    d2 = df.copy()
    d2["label"], d2["realized_ret"], d2["time_to_hit"] = label, ret, time_to_hit
    d2["sl_dist"], d2["tp_dist"] = sl_dist, tp_dist
    return d2.loc[valid].reset_index(drop=True)


def run():
    print("Building combined dataset (LightGBM TP, meta-model probability, ATR, OHLC)...")
    df = build_dataset()

    results = []
    for threshold in THRESHOLDS:
        labeled = label_at_threshold(df, threshold)
        seq_R, seq_label = sequential_r_multiples(labeled)
        n = len(seq_R)
        print(f"\n=== meta_proba >= {threshold} ({n:,} trades) ===")
        print(f"SL={100*(seq_label==-1).mean():.1f}%  TP={100*(seq_label==1).mean():.1f}%  "
              f"E[R]={seq_R.mean():.4f}  max_streak={max_streak(seq_label==-1)}")

        # per-fold consistency check
        per_fold = []
        for fold_id in sorted(labeled["fold"].unique()):
            sub_R, sub_label = sequential_r_multiples(labeled[labeled["fold"] == fold_id])
            per_fold.append({"fold": int(fold_id), "n": len(sub_R), "E_R": float(sub_R.mean())})
            print(f"  fold {fold_id} ({FOLDS[fold_id]['test_start']}): n={len(sub_R)} E[R]={sub_R.mean():.4f}")

        fs = np.linspace(0.001, 0.10, 200)
        growths = [np.mean(np.log(1 + f * seq_R)) for f in fs]
        best_f = float(fs[int(np.argmax(growths))])
        growth_at_5pct = float(np.mean(np.log(1 + 0.05 * seq_R)))

        rng = np.random.default_rng(0)
        boot = np.array([np.mean(np.log(1 + 0.05 * rng.choice(seq_R, size=n, replace=True)))
                          for _ in range(N_BOOTSTRAP)])
        p_positive = float((boot > 0).mean())

        equity = np.cumprod(1 + 0.05 * seq_R)
        running_max = np.maximum.accumulate(equity)
        max_dd = float((equity - running_max).min() / running_max[np.argmin(equity - running_max)])

        row = {
            "meta_proba_threshold": threshold, "n_trades": n,
            "sl_rate": float((seq_label == -1).mean()), "tp_rate": float((seq_label == 1).mean()),
            "arithmetic_E_R": float(seq_R.mean()), "max_consecutive_sl_streak": int(max_streak(seq_label == -1)),
            "growth_optimal_fraction": best_f, "growth_at_5pct_risk": growth_at_5pct,
            "bootstrap_P_growth_positive_at_5pct": p_positive,
            "final_equity_multiple_5pct_no_costs": float(equity[-1]),
            "max_drawdown_5pct_no_costs": max_dd,
            "per_fold": per_fold,
        }
        results.append(row)
        print(f"growth-optimal-f={best_f*100:.2f}%  growth@5%={growth_at_5pct:.6f}  "
              f"P(growth@5%>0)={p_positive*100:.1f}%  max_DD@5%={max_dd*100:.1f}%")

    out_path = Path(__file__).resolve().parent / "meta_labeled_backtest_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    run()
