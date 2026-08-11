from __future__ import annotations

import pytest

from volley_forecast.analysis import build_athlete_analysis


def test_build_athlete_analysis(demo_frame) -> None:
    result = build_athlete_analysis(demo_frame, recent_window=5)
    assert result["athlete"]["player_id"] == "demo-001"
    assert result["sample"]["participating_matches"] == len(demo_frame)
    assert result["sample"]["recent_window"] == 5
    assert result["targets"]["total_points"]["recent_mean"] is not None
    assert result["opponents"]
    assert result["rest_effects"]


def test_analysis_requires_valid_window(demo_frame) -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_athlete_analysis(demo_frame, recent_window=1)
