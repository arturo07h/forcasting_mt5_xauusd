"""Meta-labeling (López de Prado): a secondary binary classifier that predicts whether
a triple-barrier trade will actually hit TP (vs SL/timeout), trained on the SAME
features as the primary quantile model. This is a more principled selectivity filter
than the earlier conviction-by-predicted-magnitude proxy in
backtesting/model_conditioned_backtest.py — instead of inferring "is this a good setup"
from the size of the predicted move, it directly learns the conditions under which the
ATR-SL / vol-scaled-TP structure historically succeeds.

Trained on labeling/build_labels.py's placeholder-barrier outcomes (not the primary
model's own OOS predictions) to sidestep a nested-walk-forward requirement — the
placeholder rule shares the same ATR-SL / vol-scaled-TP structure as the real backtest,
so what the meta-model learns about *when that structure succeeds* should transfer
reasonably to gating entries that use the primary model's real TP quantile. Documented
simplification, not free of assumptions.

Run: ./.venv/bin/python3 models/training/train_meta_label.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from config.settings import DATA_PROCESSED_DIR, PROJECT_ROOT
from validation.dataset import load_modeling_dataset, feature_cols
from validation.walk_forward import FOLDS, fold_masks

PARAMS = dict(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    min_child_samples=200, subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)


def load_triple_barrier_labels() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "processed" / "labels" / "symbol=XAUUSD").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "label"]]


def run(profile: str = "core"):
    print("Loading features and triple-barrier outcomes...")
    feat = load_modeling_dataset(profile)
    cols = feature_cols(profile)
    tb = load_triple_barrier_labels()

    feat["time"] = pd.to_datetime(feat["time"], utc=True)
    df = feat.merge(tb, on="time", how="inner")
    df["y_meta"] = (df["label"] == 1).astype(int)
    print(f"  {len(df):,} rows, base TP rate: {df['y_meta'].mean()*100:.1f}%")

    all_preds = []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(df["time"], fold)
        X_train, y_train = df.loc[train_mask, cols], df.loc[train_mask, "y_meta"]
        X_test = df.loc[test_mask, cols]
        if len(X_test) == 0:
            continue
        print(f"Fold {i}: train={len(X_train):,} test={len(X_test):,} "
              f"(train TP rate={y_train.mean()*100:.1f}%)")

        model = lgb.LGBMClassifier(objective="binary", **PARAMS)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        fold_preds = pd.DataFrame({
            "time": df.loc[test_mask, "time"].values, "fold": i, "meta_proba": proba,
        })
        all_preds.append(fold_preds)

    preds_df = pd.concat(all_preds, ignore_index=True)
    out_path = Path(__file__).resolve().parent.parent / "checkpoints" / f"meta_label_{profile}_predictions.parquet"
    preds_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"meta_proba distribution:\n{preds_df['meta_proba'].describe()}")
    return preds_df


if __name__ == "__main__":
    run()
