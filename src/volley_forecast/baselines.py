"""Simple one-step-ahead baselines for honest model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def last_value_predictions(
    frame: pd.DataFrame, targets: list[str], label_positions: np.ndarray
) -> np.ndarray:
    predictions = []
    for position in label_positions:
        if position <= 0:
            raise ValueError("label positions must have a preceding row")
        predictions.append(frame.iloc[int(position) - 1][targets].to_numpy(dtype=float))
    return np.asarray(predictions, dtype=float)


def rolling_mean_predictions(
    frame: pd.DataFrame,
    targets: list[str],
    label_positions: np.ndarray,
    *,
    window: int = 3,
) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")
    predictions = []
    for position in label_positions:
        start = max(0, int(position) - window)
        history = frame.iloc[start : int(position)][targets].apply(pd.to_numeric, errors="coerce")
        predictions.append(history.mean().fillna(0.0).to_numpy(dtype=float))
    return np.asarray(predictions, dtype=float)
