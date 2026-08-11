from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from volley_forecast.features import engineer_features
from volley_forecast.inference import ForecastEngine
from volley_forecast.schema import DEFAULT_CONTEXT_FEATURES, DEFAULT_HISTORY_FEATURES, TARGET_COLUMNS


class FakeModel:
    def predict(self, inputs, verbose=0):
        return np.zeros((inputs["history"].shape[0], len(TARGET_COLUMNS)), dtype=float)

    def __call__(self, inputs, training=False):
        return np.zeros((inputs["history"].shape[0], len(TARGET_COLUMNS)), dtype=float)


def test_forecast_engine_with_fake_model(demo_frame) -> None:
    engineered = engineer_features(demo_frame)
    engine = ForecastEngine.__new__(ForecastEngine)
    engine.model_dir = Path("unused")
    engine.metadata = {
        "athlete": {"player_id": "demo-001", "player_name": "Demo Athlete"},
        "targets": TARGET_COLUMNS,
        "history_features": DEFAULT_HISTORY_FEATURES,
        "context_features": DEFAULT_CONTEXT_FEATURES,
        "model_config": {
            "history_steps": 5,
            "include_dnp": False,
            "interval_lower_quantile": 0.1,
            "interval_upper_quantile": 0.9,
        },
        "promotion": {"status": "promoted"},
    }
    engine.scalers = {
        "history": StandardScaler().fit(engineered[DEFAULT_HISTORY_FEATURES].fillna(0).to_numpy()),
        "context": StandardScaler().fit(engineered[DEFAULT_CONTEXT_FEATURES].fillna(0).to_numpy()),
        "target": StandardScaler().fit(engineered[TARGET_COLUMNS].fillna(0).to_numpy()),
    }
    engine.model = FakeModel()

    result = engine.forecast(
        demo_frame,
        next_date=date(2026, 8, 20),
        opponent="Italy",
        mc_samples=10,
    )
    assert result["next_match"]["opponent"] == "Italy"
    assert set(result["predictions"]) == set(TARGET_COLUMNS)
    assert result["predictions"]["total_points"]["point"] >= 0
