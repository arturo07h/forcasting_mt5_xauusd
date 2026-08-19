"""Trains LightGBM for additional TP-candidate quantiles (0.75, 0.80) — not part of the
original 5-quantile comparison, added to test whether a less extreme "eat the whole pie"
target trades some capture for enough hit-rate to matter more for Kelly growth, since SL
is always exactly -1R and growth in that regime is hit-rate-dominated.

Run: ./.venv/bin/python3 models/training/train_lightgbm_extra_quantiles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import lightgbm as lgb

from validation.dataset import load_modeling_dataset, feature_cols
from validation.walk_forward import FOLDS, fold_masks
from models.training.train_lightgbm import PARAMS

EXTRA_QUANTILES = [0.75, 0.80]


def run(profile: str = "core"):
    df = load_modeling_dataset(profile)
    cols = feature_cols(profile)
    print(f"{len(df):,} rows, {len(cols)} features")

    all_preds = []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(df["time"], fold)
        X_train, y_train = df.loc[train_mask, cols], df.loc[train_mask, "target_fwd_ret_12"]
        X_test, y_test = df.loc[test_mask, cols], df.loc[test_mask, "target_fwd_ret_12"]
        if len(X_test) == 0:
            continue
        print(f"Fold {i}: train={len(X_train):,} test={len(X_test):,}")

        fold_preds = pd.DataFrame({"time": df.loc[test_mask, "time"].values, "fold": i})
        for q in EXTRA_QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            model.fit(X_train, y_train)
            fold_preds[f"pred_q{q}"] = model.predict(X_test)
        all_preds.append(fold_preds)

    preds_df = pd.concat(all_preds, ignore_index=True)
    out_path = Path(__file__).resolve().parent.parent / "checkpoints" / f"lightgbm_{profile}_extra_quantiles.parquet"
    preds_df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    return preds_df


if __name__ == "__main__":
    run()
