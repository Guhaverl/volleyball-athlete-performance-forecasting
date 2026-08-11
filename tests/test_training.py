from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from volley_forecast.config import ModelConfig
from volley_forecast.training import train_model


class FakeModel:
    def fit(self, inputs, targets, **kwargs):
        assert kwargs["shuffle"] is False
        assert inputs["history"].shape[0] == targets.shape[0]
        return SimpleNamespace(history={"loss": [1.0], "val_loss": [1.1], "mae": [0.8]})

    def predict(self, inputs, verbose=0):
        return np.zeros((inputs["history"].shape[0], 4), dtype=np.float32)

    def save(self, path):
        path.write_text("fake keras model", encoding="utf-8")


class FakeCallbacks:
    class EarlyStopping:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ReduceLROnPlateau:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class FakeTensorFlow:
    __version__ = "test-double"
    keras = SimpleNamespace(callbacks=FakeCallbacks)


def test_train_model_writes_complete_bundle(tmp_path, demo_frame, monkeypatch) -> None:
    data_path = tmp_path / "matches.csv"
    demo_frame.to_csv(data_path, index=False)
    model_dir = tmp_path / "model"

    monkeypatch.setattr("volley_forecast.training.require_tensorflow", lambda: FakeTensorFlow())
    monkeypatch.setattr("volley_forecast.training.set_global_seed", lambda seed: None)
    monkeypatch.setattr("volley_forecast.training.build_lstm_model", lambda **kwargs: FakeModel())

    result = train_model(
        data_path=data_path,
        model_dir=model_dir,
        config=ModelConfig(epochs=2, patience=1, min_matches=20),
    )

    expected = {
        "model.keras",
        "scalers.joblib",
        "training_history.csv",
        "metadata.json",
        "metrics.json",
        "model_card.md",
    }
    assert expected.issubset({path.name for path in model_dir.iterdir()})
    assert result["metadata"]["sample_counts"]["test"] > 0
    assert result["metadata"]["promotion"]["status"] in {
        "promoted",
        "baseline_preferred",
    }
    assert result["metrics"]["tensorflow_lstm"]["n_samples"] > 0
