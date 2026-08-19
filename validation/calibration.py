"""Shared evaluation metrics for quantile forecasts — used identically across every
model family so comparisons are apples-to-apples. Pinball loss alone isn't enough (a
model can have good average pinball loss and still be badly miscalibrated in the
tails) — calibration (empirical coverage vs nominal) is checked separately, per the
Phase 2 validation plan.
"""
import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def enforce_monotonic_quantiles(preds: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Sorts predicted quantiles row-wise so a lower quantile is never predicted above
    a higher one (quantile crossing) — a simple post-hoc fix, not a training-time
    constraint.
    """
    qs = sorted(preds.keys())
    stacked = np.column_stack([preds[q] for q in qs])
    stacked.sort(axis=1)
    return {q: stacked[:, i] for i, q in enumerate(qs)}


def empirical_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of true values at or below the predicted quantile — should equal the
    quantile level q if the model is well calibrated.
    """
    return float(np.mean(y_true <= y_pred))


def evaluate_quantile_predictions(y_true: np.ndarray, preds: dict[float, np.ndarray],
                                   fold: int, model_name: str) -> pd.DataFrame:
    preds = enforce_monotonic_quantiles(preds)
    rows = []
    for q, y_pred in preds.items():
        rows.append({
            "model": model_name,
            "fold": fold,
            "quantile": q,
            "pinball_loss": pinball_loss(y_true, y_pred, q),
            "empirical_coverage": empirical_coverage(y_true, y_pred),
            "coverage_error": empirical_coverage(y_true, y_pred) - q,
        })
    return pd.DataFrame(rows)
