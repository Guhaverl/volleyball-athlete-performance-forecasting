from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from volley_forecast.exceptions import DataContractError
from volley_forecast.features import build_next_context, engineer_features
from volley_forecast.validation import validate_canonical_frame


def test_validation_excludes_dnp_and_duplicates(demo_frame: pd.DataFrame) -> None:
    frame = demo_frame.head(6).copy()
    frame.loc[1, "participated"] = False
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    cleaned = validate_canonical_frame(frame, include_dnp=False)
    assert len(cleaned) == 5
    assert cleaned["match_date"].is_monotonic_increasing


def test_validation_requires_contract() -> None:
    with pytest.raises(DataContractError, match="Missing canonical columns"):
        validate_canonical_frame(pd.DataFrame({"match_date": ["2025-01-01"]}))


def test_opponent_feature_is_shifted_without_label_leakage(demo_frame: pd.DataFrame) -> None:
    frame = demo_frame.head(3).copy()
    frame["opponent"] = ["Italy", "Brazil", "Italy"]
    frame["total_points"] = [10, 20, 30]
    engineered = engineer_features(frame)
    assert engineered.loc[2, "opponent_prior_total_points_mean"] == 10

    context = build_next_context(frame, next_date=date(2026, 1, 1), opponent="Italy")
    assert context["opponent_prior_total_points_mean"] == 20
    assert context["days_rest"] <= 60


def test_validation_parses_string_booleans(demo_frame: pd.DataFrame) -> None:
    frame = demo_frame.head(2).copy()
    frame["participated"] = ["False", "True"]
    cleaned = validate_canonical_frame(frame, include_dnp=False)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["participated"]
