"""GARCH(1,1)-skewt baseline across the same 6 walk-forward folds as LightGBM.

Run: ./.venv/bin/python3 models/training/train_garch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from validation.dataset import load_modeling_dataset, QUANTILES, _load_symbol
from validation.walk_forward import FOLDS, fold_masks
from validation.calibration import evaluate_quantile_predictions
from models.architectures.garch import fit_and_forecast, merge_onto_m5


def run():
    print("Loading target dataset and raw XAUUSD M5...")
    df = load_modeling_dataset("core")[["time", "target_fwd_ret_12"]]
    xau = _load_symbol("XAUUSD", "M5")
    xau["mid_close"] = (xau["bid_close"] + xau["ask_close"]) / 2

    all_metrics = []
    all_preds = []
    for i, fold in enumerate(FOLDS):
        train_end = pd.Timestamp(fold["train_end"], tz="UTC")
        test_end = pd.Timestamp(fold["test_end"], tz="UTC")
        print(f"\nFold {i}: fitting GARCH on daily returns < {fold['train_end']}...")

        daily_forecast = fit_and_forecast(xau, train_end, test_end, QUANTILES)
        if daily_forecast.empty:
            print("  no test data, skipping")
            continue

        _, test_mask = fold_masks(df["time"], fold)
        test_df = df.loc[test_mask].reset_index(drop=True)
        merged = merge_onto_m5(test_df["time"], daily_forecast, QUANTILES).reset_index(drop=True)
        valid = merged.notna().all(axis=1)
        y_test = test_df.loc[valid, "target_fwd_ret_12"].values
        preds = {q: merged.loc[valid, f"pred_q{q}"].values for q in QUANTILES}

        print(f"  {valid.sum():,} / {len(test_df):,} test bars scored")
        metrics = evaluate_quantile_predictions(y_test, preds, fold=i, model_name="garch")
        all_metrics.append(metrics)
        print(metrics[["quantile", "pinball_loss", "empirical_coverage", "coverage_error"]]
              .to_string(index=False))

        fold_preds = pd.DataFrame({"time": test_df.loc[valid, "time"].values, "fold": i, "y_true": y_test})
        for q in QUANTILES:
            fold_preds[f"pred_q{q}"] = preds[q]
        all_preds.append(fold_preds)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    metrics_df.to_parquet(out_dir / "garch_core_metrics.parquet", index=False)
    preds_df.to_parquet(out_dir / "garch_core_predictions.parquet", index=False)

    print("\n=== Summary (mean pinball loss by quantile, across folds) ===")
    print(metrics_df.groupby("quantile")[["pinball_loss", "coverage_error"]].mean())
    return metrics_df, preds_df


if __name__ == "__main__":
    run()
