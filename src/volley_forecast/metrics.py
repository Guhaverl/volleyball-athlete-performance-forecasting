"""Regression metrics with target-level and macro summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    if truth.ndim != 2 or truth.shape[1] != len(targets):
        raise ValueError("Expected two-dimensional arrays matching target count")

    errors = pred - truth
    absolute = np.abs(errors)
    squared = errors**2
    denominator = np.abs(truth) + np.abs(pred)
    smape = np.divide(
        2.0 * absolute,
        denominator,
        out=np.zeros_like(absolute),
        where=denominator > 0,
    )

    by_target: dict[str, dict[str, float]] = {}
    for index, target in enumerate(targets):
        by_target[target] = {
            "mae": float(absolute[:, index].mean()),
            "rmse": float(math.sqrt(squared[:, index].mean())),
            "smape": float(smape[:, index].mean()),
        }
    return {
        "n_samples": int(truth.shape[0]),
        "mae_macro": float(np.mean([item["mae"] for item in by_target.values()])),
        "rmse_macro": float(np.mean([item["rmse"] for item in by_target.values()])),
        "smape_macro": float(np.mean([item["smape"] for item in by_target.values()])),
        "by_target": by_target,
    }
