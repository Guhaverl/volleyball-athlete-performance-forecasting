from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volley_forecast.config import ModelConfig, load_athlete_config, load_model_config
from volley_forecast.demo import generate_demo_frame, write_demo_csv
from volley_forecast.schema import CANONICAL_COLUMNS


def test_load_example_configs() -> None:
    athlete = load_athlete_config(Path("config/athlete.example.yaml"))
    model = load_model_config(Path("config/model.example.yaml"))
    assert athlete.player_id == "163098"
    assert len(athlete.sources) == 3
    assert model.history_steps == 5


def test_model_config_rejects_bad_split() -> None:
    with pytest.raises(ValueError, match="leave at least"):
        ModelConfig(train_fraction=0.8, val_fraction=0.18)


def test_demo_data_is_deterministic_and_canonical(tmp_path) -> None:
    first = generate_demo_frame(matches=24, seed=3)
    second = generate_demo_frame(matches=24, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == CANONICAL_COLUMNS
    output = write_demo_csv(tmp_path / "demo.csv", matches=24, seed=3)
    assert output.exists()
    assert len(pd.read_csv(output)) == 24
