"""Parser for server-rendered Volleyball World player match tables."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from volley_forecast.exceptions import SourceAdapterError
from volley_forecast.schema import CANONICAL_COLUMNS, NUMERIC_STAT_COLUMNS

_TABLE_OUTPUTS: dict[str, list[str]] = {
    "scoring": ["total_points", "attack_points", "block_points", "serve_points"],
    "attack": [
        "attack_points_detail",
        "attack_errors",
        "attack_attempts",
        "attack_average_per_match",
        "attack_success_pct",
        "attack_total_actions",
    ],
    "block": [
        "block_points_detail",
        "block_errors",
        "block_rebounds",
        "block_average_per_match",
        "block_efficiency_pct",
        "block_total_actions",
    ],
    "serve": [
        "serve_points_detail",
        "serve_errors",
        "serve_attempts",
        "serve_average_per_match",
        "serve_success_pct",
        "serve_total_actions",
    ],
    "reception": [
        "reception_successful",
        "reception_errors",
        "reception_attempts",
        "reception_average_per_match",
        "reception_success_pct",
        "reception_total_actions",
    ],
    "dig": [
        "digs",
        "dig_errors",
        "dig_receptions",
        "dig_average_per_match",
        "dig_success_pct",
        "dig_total_actions",
    ],
    "set": [
        "set_successful",
        "set_errors",
        "set_attempts",
        "set_average_per_match",
        "set_success_pct",
        "set_total_actions",
    ],
}


def _normalize_header(value: Any) -> str:
    if isinstance(value, tuple):
        value = " ".join(str(item) for item in value if str(item) != "nan")
    text = str(value).strip().lower()
    text = text.replace("%", " pct ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if "team_a" in text or text.startswith("team_a"):
        return "team_a"
    if "team_b" in text or text.startswith("team_b"):
        return "team_b"
    if text.startswith("date"):
        return "match_date"
    return text


def _classify_table(columns: list[str]) -> str | None:
    searchable = " ".join(columns)
    if "match_date" not in columns:
        return None
    if all(
        fragment in searchable for fragment in ("attack_points", "block_points", "serve_points")
    ):
        return "scoring"
    if "rebounds" in searchable or "stuff_blocks" in searchable:
        return "block"
    if "serve_points" in searchable:
        return "serve"
    if "great_save" in searchable or re.search(r"\bdigs?\b", searchable):
        return "dig"
    if "running_sets" in searchable or "set_success" in searchable:
        return "set"
    if "successful" in searchable or "succesful" in searchable:
        return "reception"
    if "shots" in searchable or "points_attacks" in searchable:
        return "attack"
    return None


def _clean_team(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\s+[A-Z]{3}$", "", text).strip()
    match = re.match(r"^(.+?)([A-Z]{3})$", text)
    if match and len(match.group(1)) > 2:
        text = match.group(1).strip()
    return text or None


def _numeric(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "").replace(",", "")
        if value == "":
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_profile(html: str) -> dict[str, str | None]:
    soup = BeautifulSoup(html, "lxml")
    heading = soup.find("h1")
    player_name = heading.get_text(" ", strip=True) if heading else None

    def value_after_heading(label: str) -> str | None:
        node = soup.find(
            ["h1", "h2", "h3", "h4"],
            string=lambda value: (
                isinstance(value, str) and value.strip().casefold() == label.casefold()
            ),
        )
        if node is None:
            return None
        candidate = node.find_next(["h1", "h2", "h3", "h4", "p", "span"])
        while candidate is not None:
            text = candidate.get_text(" ", strip=True)
            if text and text.casefold() != label.casefold():
                return text
            candidate = candidate.find_next(["h1", "h2", "h3", "h4", "p", "span"])
        return None

    return {
        "player_name": player_name,
        "team_name": value_after_heading("Team"),
        "position": value_after_heading("Position"),
    }


def _table_to_canonical(table: pd.DataFrame, table_type: str) -> pd.DataFrame:
    table = table.copy()
    table.columns = [_normalize_header(column) for column in table.columns]
    if not {"team_a", "team_b", "match_date"}.issubset(table.columns):
        raise SourceAdapterError(f"{table_type} table is missing team/date columns")

    stat_columns = [
        column for column in table.columns if column not in {"team_a", "team_b", "match_date"}
    ]
    output_columns = _TABLE_OUTPUTS[table_type]
    rename = {
        input_name: output_name
        for input_name, output_name in zip(stat_columns, output_columns, strict=False)
    }
    keep = ["team_a", "team_b", "match_date", *rename.keys()]
    result = table[keep].rename(columns=rename)
    result["team_a"] = result["team_a"].map(_clean_team)
    result["team_b"] = result["team_b"].map(_clean_team)
    result["match_date"] = pd.to_datetime(
        result["match_date"], errors="coerce", dayfirst=True
    ).dt.date.astype("string")
    for column in output_columns:
        if column in result:
            result[column] = result[column].map(_numeric)
    return result.dropna(subset=["match_date"])


def _team_equal(value: Any, expected: str | None) -> bool:
    if expected is None or value is None:
        return False
    compact_value = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    compact_expected = re.sub(r"[^a-z0-9]", "", expected.casefold())
    return compact_expected in compact_value or compact_value in compact_expected


def parse_player_profile_html(
    html: str,
    *,
    player_id: str,
    competition_slug: str,
    season: int,
    source_url: str,
    player_name: str | None = None,
    position: str | None = None,
    team_name: str | None = None,
) -> pd.DataFrame:
    """Convert all match tables on a player page into one canonical row per match."""

    profile = _extract_profile(html)
    effective_name = player_name or profile["player_name"]
    effective_position = position or profile["position"]
    effective_team = team_name or profile["team_name"]

    try:
        raw_tables = pd.read_html(StringIO(html))
    except ValueError as exc:
        raise SourceAdapterError("No HTML tables were found on the player profile") from exc

    parsed_tables: list[pd.DataFrame] = []
    for raw_table in raw_tables:
        normalized_columns = [_normalize_header(column) for column in raw_table.columns]
        table_type = _classify_table(normalized_columns)
        if table_type is None:
            continue
        parsed_tables.append(_table_to_canonical(raw_table, table_type))

    if not parsed_tables:
        raise SourceAdapterError("No supported match-stat tables were found on the player profile")

    key = ["team_a", "team_b", "match_date"]
    merged = parsed_tables[0]
    for table in parsed_tables[1:]:
        merged = merged.merge(table, on=key, how="outer")

    for summary, detail in (
        ("attack_points", "attack_points_detail"),
        ("block_points", "block_points_detail"),
        ("serve_points", "serve_points_detail"),
    ):
        if detail in merged:
            if summary not in merged:
                merged[summary] = merged[detail]
            else:
                merged[summary] = merged[summary].fillna(merged[detail])
            merged = merged.drop(columns=[detail])

    merged["player_id"] = str(player_id)
    merged["player_name"] = effective_name
    merged["position"] = effective_position
    merged["competition_slug"] = competition_slug
    merged["season"] = season
    merged["player_team"] = None
    merged["opponent"] = None

    for index, row in merged.iterrows():
        if _team_equal(row.get("team_a"), effective_team):
            merged.at[index, "player_team"] = row.get("team_a")
            merged.at[index, "opponent"] = row.get("team_b")
        elif _team_equal(row.get("team_b"), effective_team):
            merged.at[index, "player_team"] = row.get("team_b")
            merged.at[index, "opponent"] = row.get("team_a")

    for column in NUMERIC_STAT_COLUMNS:
        if column not in merged:
            merged[column] = np.nan
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    activity_columns = [
        "attack_total_actions",
        "block_total_actions",
        "serve_total_actions",
        "reception_total_actions",
        "dig_total_actions",
        "set_total_actions",
    ]
    activity = merged[activity_columns].fillna(0).sum(axis=1)
    points = (
        merged[["total_points", "attack_points", "block_points", "serve_points"]]
        .fillna(0)
        .sum(axis=1)
    )
    merged["participated"] = (activity > 0) | (points > 0)
    merged["source_type"] = "public_player_page"
    merged["source_url"] = source_url
    merged["retrieved_at"] = datetime.now(UTC).isoformat()

    for column in CANONICAL_COLUMNS:
        if column not in merged:
            merged[column] = None

    merged = merged[CANONICAL_COLUMNS]
    merged = merged.sort_values("match_date", kind="stable").reset_index(drop=True)
    return merged
