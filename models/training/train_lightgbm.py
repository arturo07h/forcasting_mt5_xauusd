"""LightGBM multi-quantile baseline — trains one model per quantile per fold. Fast
enough to run the full 6-fold walk-forward without any data/compute compromise, unlike
the DL candidates. This is also the model family closest to the existing production
pipeline, so it doubles as the "does DL actually beat what already works" baseline.

Run: ./.venv/bin/python3 models/training/train_lightgbm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from validation.dataset import load_modeling_dataset, feature_cols, QUANTILES
from validation.walk_forward import FOLDS, fold_masks
from validation.calibration import evaluate_quantile_predictions

PARAMS = dict(
    n_estimators=400,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
)


def run(profile: str = "core"):
    print(f"Loading modeling dataset (profile={profile})...")
    df = load_modeling_dataset(profile)
    cols = feature_cols(profile)
    print(f"  {len(df):,} rows, {len(cols)} features")

    all_metrics = []
    all_preds = []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(df["time"], fold)
        X_train, y_train = df.loc[train_mask, cols], df.loc[train_mask, "target_fwd_ret_12"]
        X_test, y_test = df.loc[test_mask, cols], df.loc[test_mask, "target_fwd_ret_12"]
        if len(X_test) == 0:
            continue
        print(f"\nFold {i}: train={len(X_train):,} test={len(X_test):,} "
              f"({fold['test_start']} -> {fold['test_end']})")

        preds = {}
        t0 = time.time()
        for q in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            model.fit(X_train, y_train)
            preds[q] = model.predict(X_test)
        print(f"  trained 5 quantile models in {time.time()-t0:.1f}s")

        metrics = evaluate_quantile_predictions(y_test.values, preds, fold=i, model_name="lightgbm")
        all_metrics.append(metrics)
        print(metrics[["quantile", "pinball_loss", "empirical_coverage", "coverage_error"]]
              .to_string(index=False))

        fold_preds = pd.DataFrame({"time": df.loc[test_mask, "time"].values, "fold": i,
                                    "y_true": y_test.values})
        for q in QUANTILES:
            fold_preds[f"pred_q{q}"] = preds[q]
        all_preds.append(fold_preds)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    metrics_df.to_parquet(out_dir / f"lightgbm_{profile}_metrics.parquet", index=False)
    preds_df.to_parquet(out_dir / f"lightgbm_{profile}_predictions.parquet", index=False)

    print("\n=== Summary (mean pinball loss by quantile, across folds) ===")
    print(metrics_df.groupby("quantile")[["pinball_loss", "coverage_error"]].mean())
    return metrics_df, preds_df


if __name__ == "__main__":
    run("core")
