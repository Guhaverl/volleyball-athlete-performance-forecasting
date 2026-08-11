"""Config-driven ingestion into the canonical match table."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from volley_forecast.client import VolleyballWorldClient
from volley_forecast.config import AthleteConfig, Settings
from volley_forecast.exceptions import SourceAdapterError
from volley_forecast.json_adapter import load_json_mapping, parse_player_stats_json
from volley_forecast.parser import parse_player_profile_html
from volley_forecast.schema import CANONICAL_COLUMNS
from volley_forecast.validation import validate_canonical_frame

LOGGER = logging.getLogger(__name__)


def _safe_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _write_raw_text(raw_dir: Path, name: str, text: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_raw_json(raw_dir: Path, name: str, payload: dict[str, object]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ingest_athlete(
    athlete: AthleteConfig,
    *,
    output_path: Path,
    raw_dir: Path,
    settings: Settings | None = None,
    include_dnp: bool = True,
    continue_on_error: bool = False,
) -> pd.DataFrame:
    settings = settings or Settings()
    frames: list[pd.DataFrame] = []

    with VolleyballWorldClient(settings) as client:
        for source in athlete.sources:
            player_id = source.player_id or athlete.player_id
            team_name = source.team_name or athlete.team_name
            stem = _safe_stem(f"{source.competition_slug}_{source.season}_{player_id}")
            try:
                if (
                    source.tournament_no is not None
                    and source.tournament_start is not None
                    and source.tournament_end is not None
                ):
                    context = client.fetch_tournament(
                        date.fromisoformat(source.tournament_start),
                        date.fromisoformat(source.tournament_end),
                        source.tournament_no,
                    )
                    _write_raw_json(raw_dir, f"{stem}_tournament.json", context)

                if settings.vw_player_stats_url_template:
                    if not settings.vw_player_stats_mapping:
                        raise SourceAdapterError(
                            "VW_PLAYER_STATS_MAPPING is required with VW_PLAYER_STATS_URL_TEMPLATE"
                        )
                    payload, source_url = client.fetch_configured_player_json(
                        source.competition_slug, source.season, player_id
                    )
                    _write_raw_json(raw_dir, f"{stem}_player.json", payload)
                    mapping = load_json_mapping(settings.vw_player_stats_mapping)
                    frame = parse_player_stats_json(
                        payload,
                        mapping,
                        player_id=player_id,
                        competition_slug=source.competition_slug,
                        season=source.season,
                        source_url=source_url,
                    )
                else:
                    html, source_url = client.fetch_player_profile_html(
                        source.competition_slug, source.season, player_id
                    )
                    _write_raw_text(raw_dir, f"{stem}_player.html", html)
                    frame = parse_player_profile_html(
                        html,
                        player_id=player_id,
                        competition_slug=source.competition_slug,
                        season=source.season,
                        source_url=source_url,
                        player_name=athlete.player_name,
                        position=athlete.position,
                        team_name=team_name,
                    )
                frames.append(frame)
            except Exception:
                LOGGER.exception(
                    "Failed to ingest %s season %s for player %s",
                    source.competition_slug,
                    source.season,
                    player_id,
                )
                if not continue_on_error:
                    raise

    if not frames:
        raise SourceAdapterError("No source produced athlete match rows")

    combined = pd.concat(frames, ignore_index=True)
    for column in CANONICAL_COLUMNS:
        if column not in combined:
            combined[column] = None
    combined = validate_canonical_frame(combined[CANONICAL_COLUMNS], include_dnp=include_dnp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = combined.copy()
    serializable["match_date"] = serializable["match_date"].dt.date.astype("string")
    serializable.to_csv(output_path, index=False)
    return serializable
