"""Trains the two production models on ALL available H1 history (not a walk-forward
fold — the folds were for validation; deployment uses everything through today) and
exports both to ONNX for the MQL5 EA:
  1. LightGBM p90 quantile regressor (the TP source)
  2. LightGBM meta-labeling classifier (the entry filter, threshold >= 0.40 per the backtest)

Also exports the exact feature order and a few reference rows (features + expected
prediction) to JSON, so the MQL5-side feature computation can be validated against
Python bar-for-bar before ever going live.

Run: ./.venv/bin/python3 models/training/train_production_h1.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

from config.settings import PROJECT_ROOT
from validation.dataset_h1 import load_modeling_dataset, feature_cols
from models.training.train_lightgbm import PARAMS as LGBM_PARAMS
from models.training.train_meta_label import PARAMS as META_PARAMS

TP_QUANTILE = 0.90
ONNX_DIR = PROJECT_ROOT / "onnx_export"


def load_triple_barrier_labels() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "processed" / "labels_h1" / "symbol=XAUUSD").glob("year=*/month=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "label"]]


def export_onnx(model, n_features: int, out_path: Path, is_classifier: bool):
    initial_types = [("input", FloatTensorType([1, n_features]))]  # fixed batch=1, matches single-bar EA inference
    onnx_model = convert_lightgbm(model, initial_types=initial_types, zipmap=False)
    onnx.save_model(onnx_model, str(out_path))
    print(f"  saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main():
    ONNX_DIR.mkdir(exist_ok=True)
    profile = "core"
    cols = feature_cols(profile)
    print(f"{len(cols)} features: {cols}")

    print("\nLoading full H1 dataset...")
    df = load_modeling_dataset(profile)
    print(f"  {len(df):,} rows, {df['time'].min()} -> {df['time'].max()}")

    print("\n=== Training production TP quantile model (p90) on ALL data ===")
    tp_model = lgb.LGBMRegressor(objective="quantile", alpha=TP_QUANTILE, **LGBM_PARAMS)
    tp_model.fit(df[cols], df["target_fwd_ret_6"])
    export_onnx(tp_model, len(cols), ONNX_DIR / "xauusd_h1_tp_p90.onnx", is_classifier=False)
    tp_model.booster_.save_model(str(PROJECT_ROOT / "models" / "checkpoints" / "xauusd_h1_tp_p90.txt"))
    print("  also saved native LightGBM model (used by mql5/inference_service/live_inference.py)")

    print("\n=== Training production meta-labeling classifier on ALL data ===")
    tb = load_triple_barrier_labels()
    df_meta = df.merge(tb, on="time", how="inner")
    df_meta["y_meta"] = (df_meta["label"] == 1).astype(int)
    print(f"  {len(df_meta):,} rows, base TP rate: {df_meta['y_meta'].mean()*100:.1f}%")
    meta_model = lgb.LGBMClassifier(objective="binary", **META_PARAMS)
    meta_model.fit(df_meta[cols], df_meta["y_meta"])
    export_onnx(meta_model, len(cols), ONNX_DIR / "xauusd_h1_meta_label.onnx", is_classifier=True)
    meta_model.booster_.save_model(str(PROJECT_ROOT / "models" / "checkpoints" / "xauusd_h1_meta_label.txt"))
    print("  also saved native LightGBM model")

    print("\n=== Sanity-checking ONNX outputs match the native LightGBM predictions ===")
    import onnxruntime as ort
    tp_sess = ort.InferenceSession(str(ONNX_DIR / "xauusd_h1_tp_p90.onnx"))
    meta_sess = ort.InferenceSession(str(ONNX_DIR / "xauusd_h1_meta_label.onnx"))

    sample_df = df[cols].tail(10)
    native_tp = tp_model.predict(sample_df)
    onnx_tp = np.array([tp_sess.run(None, {"input": row.reshape(1, -1).astype(np.float32)})[0][0, 0]
                         for row in sample_df.values])
    print("  TP model — native vs onnx (max abs diff):", np.max(np.abs(native_tp - onnx_tp)))

    native_meta = meta_model.predict_proba(sample_df)[:, 1]
    onnx_meta = []
    for row in sample_df.values:
        out = meta_sess.run(None, {"input": row.reshape(1, -1).astype(np.float32)})
        proba = out[1]
        onnx_meta.append(proba[0, 1] if proba.ndim == 2 else proba[0])
    onnx_meta = np.array(onnx_meta)
    print("  Meta model — native vs onnx (max abs diff):", np.max(np.abs(native_meta - onnx_meta)))

    print("\n=== Exporting reference rows for MQL5 feature-parity validation ===")
    ref_rows = df.tail(20).copy()
    ref_rows["pred_tp_p90"] = tp_model.predict(ref_rows[cols])
    ref_meta_input = df_meta[df_meta["time"].isin(ref_rows["time"])]
    reference = {
        "feature_order": cols,
        "rows": [
            {
                "time": str(row["time"]),
                "features": {c: float(row[c]) for c in cols},
                "expected_pred_tp_p90": float(row["pred_tp_p90"]),
            }
            for _, row in ref_rows.iterrows()
        ],
    }
    ref_path = ONNX_DIR / "mql5_feature_parity_reference.json"
    ref_path.write_text(json.dumps(reference, indent=2))
    print(f"  saved {ref_path}")


if __name__ == "__main__":
    main()
