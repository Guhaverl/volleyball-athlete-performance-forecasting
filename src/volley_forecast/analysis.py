"""Descriptive athlete analysis used before and alongside forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from volley_forecast.features import engineer_features
from volley_forecast.schema import TARGET_COLUMNS
from volley_forecast.validation import validate_canonical_frame


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _linear_slope(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    x = np.arange(len(numeric), dtype=float)
    return float(np.polyfit(x, numeric.to_numpy(dtype=float), 1)[0])


def build_athlete_analysis(
    frame: pd.DataFrame,
    *,
    recent_window: int = 5,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Return JSON-ready recent form, trend, opponent, and rest summaries."""

    if recent_window < 2:
        raise ValueError("recent_window must be at least two matches")
    selected_targets = targets or list(TARGET_COLUMNS)
    canonical = validate_canonical_frame(frame, include_dnp=False)
    missing = sorted(set(selected_targets) - set(canonical.columns))
    if missing:
        raise ValueError(f"Analysis targets are missing: {', '.join(missing)}")
    engineered = engineer_features(canonical)
    recent = engineered.tail(min(recent_window, len(engineered)))

    target_summary: dict[str, dict[str, float | None]] = {}
    for target in selected_targets:
        all_values = pd.to_numeric(engineered[target], errors="coerce")
        recent_values = pd.to_numeric(recent[target], errors="coerce")
        all_mean = _safe_float(all_values.mean())
        recent_mean = _safe_float(recent_values.mean())
        target_summary[target] = {
            "career_mean": all_mean,
            "career_median": _safe_float(all_values.median()),
            "career_std": _safe_float(all_values.std(ddof=1)),
            "recent_mean": recent_mean,
            "recent_vs_career": (
                float(recent_mean - all_mean)
                if recent_mean is not None and all_mean is not None
                else None
            ),
            "recent_linear_slope_per_match": _linear_slope(recent_values),
        }

    opponent_rows = (
        engineered.assign(total_points=pd.to_numeric(engineered["total_points"], errors="coerce"))
        .groupby("opponent", dropna=False)["total_points"]
        .agg(matches="count", mean="mean", median="median", maximum="max")
        .reset_index()
        .sort_values(["matches", "mean"], ascending=[False, False])
    )
    opponents = [
        {
            "opponent": None if pd.isna(row.opponent) else str(row.opponent),
            "matches": int(row.matches),
            "mean_total_points": _safe_float(row.mean),
            "median_total_points": _safe_float(row.median),
            "max_total_points": _safe_float(row.maximum),
        }
        for row in opponent_rows.itertuples(index=False)
    ]

    rest_labels = ["0-2 days", "3-5 days", "6-9 days", "10+ days"]
    rest_bands = pd.cut(
        engineered["days_rest"],
        bins=[-0.001, 2, 5, 9, np.inf],
        labels=rest_labels,
        include_lowest=True,
    )
    rest_rows = (
        engineered.assign(rest_band=rest_bands)
        .groupby("rest_band", observed=True)["total_points"]
        .agg(matches="count", mean="mean")
        .reset_index()
    )
    rest_effects = [
        {
            "rest_band": str(row.rest_band),
            "matches": int(row.matches),
            "mean_total_points": _safe_float(row.mean),
        }
        for row in rest_rows.itertuples(index=False)
    ]

    last_row = engineered.iloc[-1]
    return {
        "athlete": {
            "player_id": str(last_row.get("player_id", "")),
            "player_name": last_row.get("player_name"),
            "position": last_row.get("position"),
        },
        "sample": {
            "participating_matches": len(engineered),
            "start_date": engineered["match_date"].min().date().isoformat(),
            "end_date": engineered["match_date"].max().date().isoformat(),
            "recent_window": len(recent),
        },
        "targets": target_summary,
        "opponents": opponents,
        "rest_effects": rest_effects,
    }
