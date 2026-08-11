"""Canonical-data validation and cleaning."""

from __future__ import annotations

import logging

import pandas as pd

from volley_forecast.exceptions import DataContractError
from volley_forecast.schema import CANONICAL_COLUMNS, NUMERIC_STAT_COLUMNS, TARGET_COLUMNS

LOGGER = logging.getLogger(__name__)


def _boolean(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().casefold()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", ""}:
        return False
    raise DataContractError(f"Unsupported participated value: {value!r}")


def validate_canonical_frame(
    frame: pd.DataFrame,
    *,
    include_dnp: bool = False,
    min_matches: int | None = None,
) -> pd.DataFrame:
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"Missing canonical columns: {', '.join(missing)}")

    cleaned = frame.copy()
    cleaned["match_date"] = pd.to_datetime(cleaned["match_date"], errors="coerce")
    bad_dates = int(cleaned["match_date"].isna().sum())
    if bad_dates:
        raise DataContractError(f"Found {bad_dates} rows with invalid match_date values")

    for column in NUMERIC_STAT_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["participated"] = cleaned["participated"].map(_boolean)
    if not include_dnp:
        removed = int((~cleaned["participated"]).sum())
        if removed:
            LOGGER.info("Excluded %s non-participation rows", removed)
        cleaned = cleaned.loc[cleaned["participated"]].copy()

    target_missing = cleaned[TARGET_COLUMNS].isna().all(axis=1)
    if target_missing.any():
        LOGGER.warning("Dropping %s rows with all target stats missing", int(target_missing.sum()))
        cleaned = cleaned.loc[~target_missing].copy()

    cleaned = cleaned.sort_values("match_date", kind="stable").reset_index(drop=True)
    duplicates = cleaned.duplicated(
        subset=["player_id", "competition_slug", "season", "match_date", "team_a", "team_b"],
        keep="last",
    )
    if duplicates.any():
        LOGGER.warning("Removed %s duplicate match rows", int(duplicates.sum()))
        cleaned = cleaned.loc[~duplicates].reset_index(drop=True)

    if min_matches is not None and len(cleaned) < min_matches:
        raise DataContractError(
            f"Only {len(cleaned)} participating matches remain; at least {min_matches} are required"
        )
    return cleaned
