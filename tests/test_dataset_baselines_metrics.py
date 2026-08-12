from __future__ import annotations

import numpy as np
import pytest

from volley_forecast.baselines import last_value_predictions, rolling_mean_predictions
from volley_forecast.dataset import chronological_split, make_sequences
from volley_forecast.features import engineer_features
from volley_forecast.metrics import evaluate_regression
from volley_forecast.schema import (
    DEFAULT_CONTEXT_FEATURES,
    DEFAULT_HISTORY_FEATURES,
    TARGET_COLUMNS,
)


def test_sequence_shapes_and_chronological_split(demo_frame) -> None:
    engineered = engineer_features(demo_frame)
    bundle = make_sequences(
        engineered,
        history_steps=5,
        history_features=DEFAULT_HISTORY_FEATURES,
        context_features=DEFAULT_CONTEXT_FEATURES,
        targets=TARGET_COLUMNS,
    )
    split = chronological_split(bundle, train_fraction=0.7, val_fraction=0.15)
    assert bundle.x_history.shape == (31, 5, len(DEFAULT_HISTORY_FEATURES))
    assert bundle.x_context.shape == (31, len(DEFAULT_CONTEXT_FEATURES))
    assert split.train.label_positions.max() < split.val.label_positions.min()
    assert split.val.label_positions.max() < split.test.label_positions.min()


def test_baselines_and_metrics(demo_frame) -> None:
    engineered = engineer_features(demo_frame)
    positions = np.asarray([5, 6, 7])
    last = last_value_predictions(engineered, TARGET_COLUMNS, positions)
    rolling = rolling_mean_predictions(engineered, TARGET_COLUMNS, positions, window=3)
    truth = engineered.iloc[positions][TARGET_COLUMNS].to_numpy(dtype=float)
    result = evaluate_regression(truth, rolling, TARGET_COLUMNS)
    assert last.shape == truth.shape
    assert rolling.shape == truth.shape
    assert result["n_samples"] == 3
    assert result["mae_macro"] >= 0


def test_metric_shape_validation() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_regression(np.zeros((2, 2)), np.zeros((3, 2)), ["a", "b"])
