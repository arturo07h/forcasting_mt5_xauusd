"""Sanity-checks the 5% fixed-risk sizing against the *unconditional* (blind-entry,
every valid bar) triple-barrier labels — i.e. "what happens if there is no model at
all, just this SL/TP rule fired on every signal." This is a worst-case/no-selectivity
baseline, not a claim about the eventual model's real performance: the quantile model's
entire job is to be selective enough to beat this baseline. Run after labeling/build_labels.py.

Uses non-overlapping (sequential) trades — taking every valid bar as a separate
"trade" wildly overstates loss-streak risk, since the bot only holds one position at a
time and consecutive overlapping bars during one bad stretch aren't independent losses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT

RISK_FRACTIONS_TO_TEST = [0.05, 0.03, 0.02, 0.01, 0.005]


def load_labels() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "processed" / "labels" / "symbol=XAUUSD").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("time").reset_index(drop=True)


def sequential_r_multiples(df: pd.DataFrame, cost_frac: np.ndarray = None) -> np.ndarray:
    """cost_frac: optional per-row round-trip transaction cost, as a fraction of entry
    price, subtracted from the trade's realized return before computing R. None = no
    costs (the default used everywhere until the cost-aware backtest was added).
    """
    labels = df["label"].values
    tth = df["time_to_hit"].values
    ret = df["realized_ret"].values
    sl_dist = df["sl_dist"].values
    tp_dist = df["tp_dist"].values
    price = df["mid_close"].values
    n = len(df)
    cost = cost_frac if cost_frac is not None else np.zeros(n)

    # a handful of bars have near-zero ATR (thin early-history liquidity) which blows up
    # R = ret/sl_frac toward infinity — floor sl_dist at its 1st percentile so those bars
    # don't dominate the result; this is a data-quality accommodation, not a rule change
    sl_floor = np.percentile(sl_dist / price, 1) * price
    sl_dist_safe = np.maximum(sl_dist, sl_floor)

    seq_R, seq_label = [], []
    i = 0
    while i < n:
        lab = labels[i]
        if lab == -1:
            base_ret = -sl_dist[i] / price[i]
        elif lab == 1:
            base_ret = tp_dist[i] / price[i]
        else:
            base_ret = ret[i]
        adj_ret = base_ret - cost[i]
        R = adj_ret * price[i] / sl_dist_safe[i]
        seq_R.append(R)
        seq_label.append(lab)
        i += max(int(tth[i]), 1)
    return np.array(seq_R), np.array(seq_label)


def max_streak(is_loss: np.ndarray) -> int:
    streaks, cur = [], 0
    for v in is_loss:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return max(streaks) if streaks else 0


def run():
    df = load_labels()
    seq_R, seq_label = sequential_r_multiples(df)
    n_trades = len(seq_R)
    is_loss = seq_label == -1
    streak = max_streak(is_loss)

    print(f"{n_trades:,} sequential (non-overlapping) trades over {df['time'].min()} -> {df['time'].max()}")
    print(f"SL rate: {(seq_label==-1).mean()*100:.1f}%, TP rate: {(seq_label==1).mean()*100:.1f}%, "
          f"timeout: {(seq_label==0).mean()*100:.1f}%")
    print(f"Arithmetic E[R] per trade: {seq_R.mean():.4f}")
    print(f"Max consecutive SL streak: {streak} -> {(1-0.95**streak)*100:.1f}% drawdown at 5% fixed risk if it recurred")

    results = {}
    for f in RISK_FRACTIONS_TO_TEST:
        growth_per_trade = np.mean(np.log(1 + f * seq_R))
        results[f] = growth_per_trade
        verdict = "RUIN (negative geometric growth)" if growth_per_trade < 0 else "survives"
        print(f"  risk={f*100:.1f}%: E[log(1+f*R)]={growth_per_trade:.6f}/trade -> {verdict}")

    fs = np.linspace(0.001, 0.06, 120)
    growths = [np.mean(np.log(1 + f * seq_R)) for f in fs]
    best_f = float(fs[int(np.argmax(growths))])
    print(f"\nGrowth-optimal (Kelly-style) fixed fraction for THIS unconditional baseline: {best_f*100:.2f}%")
    print("(this is the blind-entry baseline the trained model needs to beat, not a recommended risk level)")

    summary = {
        "n_sequential_trades": int(n_trades),
        "sl_rate": float((seq_label == -1).mean()),
        "tp_rate": float((seq_label == 1).mean()),
        "timeout_rate": float((seq_label == 0).mean()),
        "arithmetic_mean_R": float(seq_R.mean()),
        "max_consecutive_sl_streak": int(streak),
        "drawdown_at_max_streak_5pct_risk": float(1 - 0.95 ** streak),
        "geometric_growth_per_trade_by_risk_fraction": {str(f): float(g) for f, g in results.items()},
        "kelly_optimal_fixed_fraction": best_f,
        "note": ("Unconditional/blind-entry baseline — every valid bar taken as a trade, no "
                 "model selectivity, no transaction costs. This is a worst-case floor to compare "
                 "the trained quantile model against, not a performance forecast."),
    }
    out_path = Path(__file__).resolve().parent / "baseline_risk_check_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return summary


if __name__ == "__main__":
    run()
