"""End-to-end TensorFlow training, evaluation, and artifact persistence."""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from volley_forecast.baselines import last_value_predictions, rolling_mean_predictions
from volley_forecast.config import ModelConfig
from volley_forecast.dataset import SequenceBundle, chronological_split, make_sequences
from volley_forecast.features import engineer_features
from volley_forecast.metrics import evaluate_regression
from volley_forecast.modeling import build_lstm_model, require_tensorflow, set_global_seed
from volley_forecast.validation import validate_canonical_frame

LOGGER = logging.getLogger(__name__)


def _scale_bundle(
    bundle: SequenceBundle,
    *,
    history_scaler: StandardScaler,
    context_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> SequenceBundle:
    sample_count, step_count, feature_count = bundle.x_history.shape
    scaled_history = history_scaler.transform(
        bundle.x_history.reshape(sample_count * step_count, feature_count)
    ).reshape(sample_count, step_count, feature_count)
    return SequenceBundle(
        x_history=scaled_history.astype(np.float32),
        x_context=context_scaler.transform(bundle.x_context).astype(np.float32),
        y=target_scaler.transform(bundle.y).astype(np.float32),
        label_positions=bundle.label_positions.copy(),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _package_versions(tf: Any) -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "tensorflow": str(tf.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _date_range(frame: pd.DataFrame, positions: np.ndarray) -> dict[str, str | None]:
    if len(positions) == 0:
        return {"start": None, "end": None}
    dates = frame.iloc[positions]["match_date"]
    return {
        "start": pd.Timestamp(dates.min()).date().isoformat(),
        "end": pd.Timestamp(dates.max()).date().isoformat(),
    }


def _render_generated_model_card(
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    status = metadata["promotion"]["status"]
    lines = [
        "# Generated Model Card",
        "",
        f"- **Created:** {metadata['created_at']}",
        f"- **Athlete:** {metadata['athlete'].get('player_name') or metadata['athlete'].get('player_id')}",
        f"- **Architecture:** compact LSTM with a next-match context branch",
        f"- **Promotion status:** `{status}`",
        f"- **History window:** {metadata['model_config']['history_steps']} matches",
        f"- **Training samples:** {metadata['sample_counts']['train']}",
        f"- **Validation samples:** {metadata['sample_counts']['validation']}",
        f"- **Test samples:** {metadata['sample_counts']['test']}",
        "",
        "## Held-out results",
        "",
        "| Candidate | Macro MAE | Macro RMSE | Macro sMAPE |",
        "|---|---:|---:|---:|",
    ]
    for candidate in ("tensorflow_lstm", "last_value", "rolling_mean"):
        result = metrics[candidate]
        lines.append(
            f"| {candidate} | {result['mae_macro']:.3f} | "
            f"{result['rmse_macro']:.3f} | {result['smape_macro']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Intended use",
            "",
            "Exploratory, athlete-specific next-match stat forecasting. Predictions are not "
            "medical, betting, selection, employment, or contract advice.",
            "",
            "## Limitations",
            "",
            "Small athlete-level samples, changing roles, injuries, roster decisions, rule changes, "
            "competition strength, and missing playing-time data can materially reduce accuracy. "
            "Retrain after meaningful role or season changes and compare every candidate with the "
            "included persistence baselines.",
            "",
        ]
    )
    return "\n".join(lines)


def train_model(
    *,
    data_path: Path,
    model_dir: Path,
    config: ModelConfig,
) -> dict[str, Any]:
    """Train one athlete-specific model and persist a complete artifact bundle."""

    tf = require_tensorflow()
    set_global_seed(config.random_seed)

    raw = pd.read_csv(data_path)
    canonical = validate_canonical_frame(
        raw,
        include_dnp=config.include_dnp,
        min_matches=config.min_matches,
    )
    engineered = engineer_features(canonical)

    configured_columns = set(config.history_features + config.context_features + config.targets)
    missing = sorted(configured_columns - set(engineered.columns))
    if missing:
        raise ValueError(f"Configured modelling columns are missing: {', '.join(missing)}")

    bundle = make_sequences(
        engineered,
        history_steps=config.history_steps,
        history_features=config.history_features,
        context_features=config.context_features,
        targets=config.targets,
    )
    split = chronological_split(
        bundle,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )

    history_scaler = StandardScaler().fit(
        split.train.x_history.reshape(-1, split.train.x_history.shape[-1])
    )
    context_scaler = StandardScaler().fit(split.train.x_context)
    target_scaler = StandardScaler().fit(split.train.y)

    train_scaled = _scale_bundle(
        split.train,
        history_scaler=history_scaler,
        context_scaler=context_scaler,
        target_scaler=target_scaler,
    )
    val_scaled = _scale_bundle(
        split.val,
        history_scaler=history_scaler,
        context_scaler=context_scaler,
        target_scaler=target_scaler,
    )
    test_scaled = _scale_bundle(
        split.test,
        history_scaler=history_scaler,
        context_scaler=context_scaler,
        target_scaler=target_scaler,
    )

    model = build_lstm_model(
        history_steps=config.history_steps,
        history_feature_count=len(config.history_features),
        context_feature_count=len(config.context_features),
        target_count=len(config.targets),
        lstm_units=config.lstm_units,
        context_units=config.context_units,
        dense_units=config.dense_units,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, config.patience // 3),
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        {"history": train_scaled.x_history, "context": train_scaled.x_context},
        train_scaled.y,
        validation_data=(
            {"history": val_scaled.x_history, "context": val_scaled.x_context},
            val_scaled.y,
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )

    prediction_scaled = model.predict(
        {"history": test_scaled.x_history, "context": test_scaled.x_context},
        verbose=0,
    )
    model_predictions = np.clip(target_scaler.inverse_transform(prediction_scaled), 0.0, None)
    y_true = split.test.y.astype(float)
    last_predictions = np.clip(
        last_value_predictions(engineered, config.targets, split.test.label_positions),
        0.0,
        None,
    )
    rolling_predictions = np.clip(
        rolling_mean_predictions(
            engineered,
            config.targets,
            split.test.label_positions,
            window=config.rolling_baseline_window,
        ),
        0.0,
        None,
    )

    metrics = {
        "tensorflow_lstm": evaluate_regression(y_true, model_predictions, config.targets),
        "last_value": evaluate_regression(y_true, last_predictions, config.targets),
        "rolling_mean": evaluate_regression(y_true, rolling_predictions, config.targets),
    }
    candidate_score = float(metrics["tensorflow_lstm"][config.promotion_metric])
    best_baseline_name = min(
        ("last_value", "rolling_mean"),
        key=lambda name: float(metrics[name][config.promotion_metric]),
    )
    best_baseline_score = float(metrics[best_baseline_name][config.promotion_metric])
    promoted = candidate_score < best_baseline_score
    improvement_pct = (
        100.0 * (best_baseline_score - candidate_score) / best_baseline_score
        if best_baseline_score > 0
        else 0.0
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")
    joblib.dump(
        {
            "history": history_scaler,
            "context": context_scaler,
            "target": target_scaler,
        },
        model_dir / "scalers.joblib",
    )
    pd.DataFrame(history.history).to_csv(model_dir / "training_history.csv", index=False)

    athlete_row = canonical.iloc[-1]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "data_path": str(data_path),
        "athlete": {
            "player_id": str(athlete_row.get("player_id", "")),
            "player_name": athlete_row.get("player_name"),
            "position": athlete_row.get("position"),
        },
        "targets": config.targets,
        "history_features": config.history_features,
        "context_features": config.context_features,
        "model_config": config.model_dump(mode="json"),
        "sample_counts": {
            "canonical_matches": len(canonical),
            "sequences": len(bundle),
            "train": len(split.train),
            "validation": len(split.val),
            "test": len(split.test),
        },
        "date_ranges": {
            "train": _date_range(engineered, split.train.label_positions),
            "validation": _date_range(engineered, split.val.label_positions),
            "test": _date_range(engineered, split.test.label_positions),
        },
        "promotion": {
            "status": "promoted" if promoted else "baseline_preferred",
            "metric": config.promotion_metric,
            "candidate_score": candidate_score,
            "best_baseline": best_baseline_name,
            "best_baseline_score": best_baseline_score,
            "improvement_pct": improvement_pct,
        },
        "versions": _package_versions(tf),
    }

    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    (model_dir / "model_card.md").write_text(
        _render_generated_model_card(metadata=metadata, metrics=metrics), encoding="utf-8"
    )
    LOGGER.info("Saved model bundle to %s", model_dir)
    return {"metadata": metadata, "metrics": metrics}
