"""Load a saved athlete model and produce next-match forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from volley_forecast.features import build_next_context, engineer_features
from volley_forecast.modeling import monte_carlo_predictions, require_tensorflow
from volley_forecast.validation import validate_canonical_frame


@dataclass
class ForecastEngine:
    model_dir: Path

    def __post_init__(self) -> None:
        tf = require_tensorflow()
        metadata_path = self.model_dir / "metadata.json"
        scalers_path = self.model_dir / "scalers.joblib"
        model_path = self.model_dir / "model.keras"
        if not metadata_path.exists() or not scalers_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Incomplete model bundle in {self.model_dir}; expected model.keras, "
                "metadata.json, and scalers.joblib"
            )
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.scalers: dict[str, Any] = joblib.load(scalers_path)
        self.model: Any = tf.keras.models.load_model(model_path)

    def forecast(
        self,
        history: pd.DataFrame,
        *,
        next_date: date,
        opponent: str | None = None,
        mc_samples: int = 100,
    ) -> dict[str, Any]:
        model_config = self.metadata["model_config"]
        history_steps = int(model_config["history_steps"])
        canonical = validate_canonical_frame(
            history,
            include_dnp=bool(model_config.get("include_dnp", False)),
            min_matches=history_steps,
        )
        engineered = engineer_features(canonical)

        history_features = list(self.metadata["history_features"])
        context_features = list(self.metadata["context_features"])
        targets = list(self.metadata["targets"])
        required = set(history_features + targets)
        missing = sorted(required - set(engineered.columns))
        if missing:
            raise ValueError(f"Forecast history is missing required columns: {', '.join(missing)}")

        history_values = (
            engineered.tail(history_steps)[history_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        context_map = build_next_context(canonical, next_date=next_date, opponent=opponent)
        missing_context = sorted(set(context_features) - set(context_map))
        if missing_context:
            raise ValueError(
                f"No next-match builder is defined for: {', '.join(missing_context)}"
            )
        context_values = np.asarray(
            [[float(context_map[name]) for name in context_features]], dtype=np.float32
        )

        history_scaler = self.scalers["history"]
        context_scaler = self.scalers["context"]
        target_scaler = self.scalers["target"]
        scaled_history = history_scaler.transform(history_values).reshape(
            1, history_steps, len(history_features)
        )
        scaled_context = context_scaler.transform(context_values)

        point_scaled = self.model.predict(
            {"history": scaled_history, "context": scaled_context}, verbose=0
        )
        point = np.clip(target_scaler.inverse_transform(point_scaled)[0], 0.0, None)

        samples_scaled = monte_carlo_predictions(
            self.model,
            x_history=scaled_history.astype(np.float32),
            x_context=scaled_context.astype(np.float32),
            samples=mc_samples,
        )
        samples_flat = samples_scaled.reshape(-1, len(targets))
        samples = target_scaler.inverse_transform(samples_flat).reshape(mc_samples, 1, -1)
        samples = np.clip(samples[:, 0, :], 0.0, None)
        lower_q = float(model_config["interval_lower_quantile"])
        upper_q = float(model_config["interval_upper_quantile"])
        lower = np.quantile(samples, lower_q, axis=0)
        upper = np.quantile(samples, upper_q, axis=0)

        predictions = {
            target: {
                "point": float(point[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "rounded_point": int(round(float(point[index]))),
            }
            for index, target in enumerate(targets)
        }
        return {
            "athlete": self.metadata.get("athlete", {}),
            "next_match": {
                "date": next_date.isoformat(),
                "opponent": opponent,
                "context": context_map,
            },
            "interval_quantiles": {"lower": lower_q, "upper": upper_q},
            "predictions": predictions,
            "model_promotion": self.metadata.get("promotion", {}),
        }
