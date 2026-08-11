"""Leakage-safe feature engineering for athlete match histories."""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def _opponent_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return re.sub(r"[^a-z0-9]", "", str(value).casefold()) or "unknown"


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values("match_date", kind="stable").reset_index(drop=True)
    result["match_date"] = pd.to_datetime(result["match_date"], errors="raise")
    result["days_rest"] = (
        result["match_date"].diff().dt.total_seconds().div(86400).clip(lower=0, upper=60).fillna(0)
    )
    day_of_year = result["match_date"].dt.dayofyear.astype(float)
    result["season_day_sin"] = np.sin(2 * math.pi * day_of_year / 366.0)
    result["season_day_cos"] = np.cos(2 * math.pi * day_of_year / 366.0)
    result["match_index"] = np.arange(len(result), dtype=float)

    result["opponent_key"] = result["opponent"].map(_opponent_key)
    overall_prior = result["total_points"].shift(1).expanding(min_periods=1).mean()
    opponent_prior = result.groupby("opponent_key", dropna=False)["total_points"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    result["opponent_prior_total_points_mean"] = opponent_prior.fillna(overall_prior).fillna(0.0)
    return result


def build_next_context(
    historical_frame: pd.DataFrame,
    *,
    next_date: date,
    opponent: str | None,
) -> dict[str, float]:
    if historical_frame.empty:
        raise ValueError("historical_frame cannot be empty")
    history = engineer_features(historical_frame)
    last_date = history["match_date"].max().date()
    days_rest = float(max((next_date - last_date).days, 0))
    day_of_year = float(next_date.timetuple().tm_yday)
    opponent_key = _opponent_key(opponent)
    prior_opponent_rows = history.loc[history["opponent_key"] == opponent_key, "total_points"]
    if prior_opponent_rows.notna().any():
        opponent_mean = float(prior_opponent_rows.mean())
    elif history["total_points"].notna().any():
        opponent_mean = float(history["total_points"].mean())
    else:
        opponent_mean = 0.0
    return {
        "days_rest": min(days_rest, 60.0),
        "season_day_sin": math.sin(2 * math.pi * day_of_year / 366.0),
        "season_day_cos": math.cos(2 * math.pi * day_of_year / 366.0),
        "opponent_prior_total_points_mean": opponent_mean,
    }
