"""TensorFlow model construction with lazy imports for lightweight core installs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from volley_forecast.exceptions import TensorFlowUnavailableError


def require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise TensorFlowUnavailableError(
            'TensorFlow is not installed. Run: python -m pip install -e ".[ml]"'
        ) from exc
    return tf


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(seed)


def build_lstm_model(
    *,
    history_steps: int,
    history_feature_count: int,
    context_feature_count: int,
    target_count: int,
    lstm_units: int,
    context_units: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
) -> Any:
    tf = require_tensorflow()
    history_input = tf.keras.layers.Input(
        shape=(history_steps, history_feature_count), name="history"
    )
    context_input = tf.keras.layers.Input(shape=(context_feature_count,), name="context")

    history_branch = tf.keras.layers.LSTM(
        lstm_units,
        dropout=dropout,
        recurrent_dropout=0.0,
        name="history_lstm",
    )(history_input)

    context_branch = tf.keras.layers.Dense(context_units, activation="relu", name="context_dense")(
        context_input
    )
    combined = tf.keras.layers.Concatenate()([history_branch, context_branch])
    combined = tf.keras.layers.Dense(dense_units, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(dropout, name="forecast_dropout")(combined)
    output = tf.keras.layers.Dense(target_count, name="stat_forecast")(combined)

    model = tf.keras.Model(
        inputs={"history": history_input, "context": context_input},
        outputs=output,
        name="athlete_stat_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def monte_carlo_predictions(
    model: Any,
    *,
    x_history: np.ndarray,
    x_context: np.ndarray,
    samples: int,
) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be positive")
    predictions = []
    for _ in range(samples):
        value = model({"history": x_history, "context": x_context}, training=True)
        predictions.append(np.asarray(value, dtype=float))
    return np.stack(predictions, axis=0)
