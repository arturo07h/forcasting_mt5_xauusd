"""H1 counterpart of train_meta_label.py.

Run: ./.venv/bin/python3 models/training/train_meta_label_h1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import lightgbm as lgb

from config.settings import PROJECT_ROOT
from validation.dataset_h1 import load_modeling_dataset, feature_cols
from validation.walk_forward_h1 import FOLDS, fold_masks
from models.training.train_meta_label import PARAMS


def load_triple_barrier_labels() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "processed" / "labels_h1" / "symbol=XAUUSD").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "label"]]


def run(profile: str = "core"):
    print("Loading H1 features and triple-barrier outcomes...")
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
        print(f"Fold {i}: train={len(X_train):,} test={len(X_test):,} (train TP rate={y_train.mean()*100:.1f}%)")

        model = lgb.LGBMClassifier(objective="binary", **PARAMS)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        fold_preds = pd.DataFrame({"time": df.loc[test_mask, "time"].values, "fold": i, "meta_proba": proba})
        all_preds.append(fold_preds)

    preds_df = pd.concat(all_preds, ignore_index=True)
    out_path = Path(__file__).resolve().parent.parent / "checkpoints" / f"meta_label_h1_{profile}_predictions.parquet"
    preds_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"meta_proba distribution:\n{preds_df['meta_proba'].describe()}")
    return preds_df


if __name__ == "__main__":
    run()
