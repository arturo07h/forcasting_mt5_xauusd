"""Trains TCN or LSTM quantile models across the 6 walk-forward folds.

Compute-budget decisions (this machine: Apple M3, 8GB RAM, no dedicated GPU — MPS
only), documented rather than silently applied:
  - training windows capped at MAX_TRAIN_WINDOWS per fold via random subsampling of
    start indices (fixed seed) when a fold's training set would otherwise produce more
    — a full epoch over fold 5's 1.3M rows takes ~9 minutes on this hardware, and this
    needs to run across 6 folds x 2 architectures
  - a chronological (not random) 5% tail of each fold's training range is held out for
    early stopping, so validation never leaks into training regardless of subsampling

Run: ./.venv/bin/python3 models/training/train_dl.py tcn
     ./.venv/bin/python3 models/training/train_dl.py lstm
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from validation.dataset import load_modeling_dataset, feature_cols, QUANTILES
from validation.walk_forward import FOLDS, fold_masks
from validation.calibration import evaluate_quantile_predictions
from models.architectures.quantile_heads import pinball_loss_multi
from models.training.sequence_dataset import SequenceDataset, standardize

SEQ_LEN = 64
MAX_TRAIN_WINDOWS = 400_000
BATCH_SIZE = 256
EPOCHS = 3
VAL_FRACTION = 0.05
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(name: str, n_features: int):
    if name == "tcn":
        from models.architectures.tcn import TCN
        return TCN(n_features=n_features, quantiles=QUANTILES)
    if name == "lstm":
        from models.architectures.lstm import LSTMQuantile
        return LSTMQuantile(n_features=n_features, quantiles=QUANTILES)
    raise ValueError(name)


def train_one_fold(model_name, X_train_full, y_train_full, X_test, y_test, n_features):
    n = len(X_train_full)
    val_start = int(n * (1 - VAL_FRACTION))
    X_tr_raw, y_tr = X_train_full[:val_start], y_train_full[:val_start]
    X_val_raw, y_val = X_train_full[val_start:], y_train_full[val_start:]

    X_tr, X_val, X_test_std = standardize(X_tr_raw, X_val_raw, X_test)

    train_ds = SequenceDataset(X_tr, y_tr, SEQ_LEN)
    val_ds = SequenceDataset(X_val, y_val, SEQ_LEN)
    test_ds = SequenceDataset(X_test_std, y_test, SEQ_LEN)

    rng = np.random.default_rng(SEED)
    if len(train_ds) > MAX_TRAIN_WINDOWS:
        idx = rng.choice(len(train_ds), size=MAX_TRAIN_WINDOWS, replace=False)
        train_ds = Subset(train_ds, idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = build_model(model_name, n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            preds = model(xb)
            loss = pinball_loss_multi(preds, yb, QUANTILES)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_loss += pinball_loss_multi(preds, yb, QUANTILES).item()
                n_val_batches += 1
        val_loss /= n_val_batches

        print(f"    epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"({time.time()-t0:.0f}s)")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets


def run(model_name: str, profile: str = "core"):
    print(f"Loading modeling dataset (profile={profile})...")
    df = load_modeling_dataset(profile)
    cols = feature_cols(profile)
    print(f"  {len(df):,} rows, {len(cols)} features, device={DEVICE}")

    all_metrics, all_preds = [], []
    for i, fold in enumerate(FOLDS):
        train_mask, test_mask = fold_masks(df["time"], fold)
        X_train = df.loc[train_mask, cols].values
        y_train = df.loc[train_mask, "target_fwd_ret_12"].values
        X_test = df.loc[test_mask, cols].values
        y_test = df.loc[test_mask, "target_fwd_ret_12"].values
        test_time = df.loc[test_mask, "time"].values
        if len(X_test) <= SEQ_LEN:
            continue

        print(f"\nFold {i} [{model_name}]: train={len(X_train):,} test={len(X_test):,} "
              f"({fold['test_start']} -> {fold['test_end']})")
        preds_arr, targets_arr = train_one_fold(model_name, X_train, y_train, X_test, y_test, len(cols))

        preds = {q: preds_arr[:, j] for j, q in enumerate(sorted(QUANTILES))}
        metrics = evaluate_quantile_predictions(targets_arr, preds, fold=i, model_name=model_name)
        all_metrics.append(metrics)
        print(metrics[["quantile", "pinball_loss", "empirical_coverage", "coverage_error"]]
              .to_string(index=False))

        # test_time is offset by SEQ_LEN-1 since the first SEQ_LEN-1 test rows can't form a full window
        fold_preds = pd.DataFrame({"time": test_time[SEQ_LEN - 1:], "fold": i, "y_true": targets_arr})
        for j, q in enumerate(sorted(QUANTILES)):
            fold_preds[f"pred_q{q}"] = preds_arr[:, j]
        all_preds.append(fold_preds)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    metrics_df.to_parquet(out_dir / f"{model_name}_{profile}_metrics.parquet", index=False)
    preds_df.to_parquet(out_dir / f"{model_name}_{profile}_predictions.parquet", index=False)

    print(f"\n=== {model_name} summary (mean pinball loss by quantile, across folds) ===")
    print(metrics_df.groupby("quantile")[["pinball_loss", "coverage_error"]].mean())
    return metrics_df, preds_df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "tcn")
