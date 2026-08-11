"""Configurable adapter for an authorized athlete-stat JSON endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from volley_forecast.exceptions import SourceAdapterError
from volley_forecast.schema import CANONICAL_COLUMNS, NUMERIC_STAT_COLUMNS


class JsonMapping(BaseModel):
    records_path: str
    profile: dict[str, str] = Field(default_factory=dict)
    fields: dict[str, str] = Field(default_factory=dict)
    constants: dict[str, Any] = Field(default_factory=dict)
    date_format: str | None = None


def load_json_mapping(path: Path) -> JsonMapping:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return JsonMapping.model_validate(payload)


def get_path(value: Any, path: str, default: Any = None) -> Any:
    current = value
    if path in {"", "."}:
        return current
    for component in path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "").replace(",", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool:
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
    raise SourceAdapterError(f"Unsupported participated value: {value!r}")


def parse_player_stats_json(
    payload: dict[str, Any],
    mapping: JsonMapping,
    *,
    player_id: str,
    competition_slug: str,
    season: int,
    source_url: str,
) -> pd.DataFrame:
    records = get_path(payload, mapping.records_path)
    if not isinstance(records, list):
        raise SourceAdapterError(
            f"JSON records_path '{mapping.records_path}' did not resolve to a list"
        )

    profile_values = {
        output_name: get_path(payload, input_path)
        for output_name, input_path in mapping.profile.items()
    }
    rows: list[dict[str, Any]] = []
    retrieved_at = datetime.now(UTC).isoformat()
    for record in records:
        if not isinstance(record, dict):
            continue
        row: dict[str, Any] = {
            "player_id": player_id,
            "competition_slug": competition_slug,
            "season": season,
            "source_type": "authorized_json",
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            **mapping.constants,
            **profile_values,
        }
        for output_name, input_path in mapping.fields.items():
            row[output_name] = get_path(record, input_path)
        rows.append(row)

    if not rows:
        raise SourceAdapterError("Authorized JSON response contained no usable match records")

    frame = pd.DataFrame(rows)
    if "match_date" not in frame.columns:
        raise SourceAdapterError("JSON mapping must populate match_date")
    frame["match_date"] = pd.to_datetime(
        frame["match_date"], format=mapping.date_format, errors="coerce", dayfirst=True
    ).dt.date.astype("string")

    for column in NUMERIC_STAT_COLUMNS:
        if column not in frame:
            frame[column] = None
        frame[column] = frame[column].map(_number)

    if "participated" not in frame:
        activity = (
            frame[NUMERIC_STAT_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .abs()
            .sum(axis=1)
        )
        frame["participated"] = activity > 0
    else:
        frame["participated"] = frame["participated"].map(_boolean)

    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[CANONICAL_COLUMNS]
