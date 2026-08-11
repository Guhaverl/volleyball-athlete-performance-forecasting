"""Runtime, ingestion, and model configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from volley_forecast.schema import (
    DEFAULT_CONTEXT_FEATURES,
    DEFAULT_HISTORY_FEATURES,
    TARGET_COLUMNS,
)


class Settings(BaseSettings):
    """Environment-backed settings for Volleyball World access."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    vw_base_url: str = "https://en.volleyballworld.com"
    vw_timeout_seconds: float = 30.0
    vw_request_interval_seconds: float = 1.0
    vw_cache_dir: Path = Path(".cache/volleyball_world")
    vw_cache_ttl_hours: float = 24.0
    vw_user_agent: str = "volleyball-athlete-forecasting/0.1 (+replace-with-contact-email)"
    vw_player_stats_url_template: str | None = None
    vw_player_stats_mapping: Path | None = Path("config/vw-player-json-mapping.example.yaml")
    model_dir: Path = Path("artifacts/model")
    log_level: str = "INFO"

    @field_validator("vw_base_url")
    @classmethod
    def strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("vw_player_stats_url_template", mode="before")
    @classmethod
    def blank_template_to_none(cls, value: Any) -> Any:
        if value is None or str(value).strip() == "":
            return None
        return str(value).strip()


class AthleteSourceConfig(BaseModel):
    competition_slug: str
    season: int
    player_id: str | None = None
    team_name: str | None = None
    tournament_no: int | None = None
    tournament_start: str | None = None
    tournament_end: str | None = None

    @model_validator(mode="after")
    def tournament_context_is_complete(self) -> "AthleteSourceConfig":
        supplied = [self.tournament_no is not None, self.tournament_start is not None, self.tournament_end is not None]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "tournament_no, tournament_start, and tournament_end must be supplied together"
            )
        return self


class AthleteConfig(BaseModel):
    player_id: str
    player_name: str | None = None
    position: str | None = None
    team_name: str | None = None
    sources: list[AthleteSourceConfig] = Field(min_length=1)


class ModelConfig(BaseModel):
    history_steps: int = Field(default=5, ge=2, le=30)
    targets: list[str] = Field(default_factory=lambda: list(TARGET_COLUMNS))
    history_features: list[str] = Field(default_factory=lambda: list(DEFAULT_HISTORY_FEATURES))
    context_features: list[str] = Field(default_factory=lambda: list(DEFAULT_CONTEXT_FEATURES))
    min_matches: int = Field(default=20, ge=12)
    include_dnp: bool = False
    train_fraction: float = Field(default=0.70, gt=0.4, lt=0.9)
    val_fraction: float = Field(default=0.15, gt=0.05, lt=0.4)
    epochs: int = Field(default=120, ge=1, le=2000)
    batch_size: int = Field(default=16, ge=1, le=1024)
    lstm_units: int = Field(default=32, ge=4, le=512)
    context_units: int = Field(default=16, ge=2, le=256)
    dense_units: int = Field(default=32, ge=4, le=512)
    dropout: float = Field(default=0.20, ge=0.0, lt=0.9)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    patience: int = Field(default=15, ge=1, le=200)
    random_seed: int = 42
    rolling_baseline_window: int = Field(default=3, ge=2, le=20)
    interval_lower_quantile: float = Field(default=0.10, ge=0.0, lt=0.5)
    interval_upper_quantile: float = Field(default=0.90, gt=0.5, le=1.0)
    promotion_metric: str = "mae_macro"

    @model_validator(mode="after")
    def validate_fractions_and_quantiles(self) -> "ModelConfig":
        if self.train_fraction + self.val_fraction >= 0.95:
            raise ValueError("train_fraction + val_fraction must leave at least 5% for test")
        if self.interval_lower_quantile >= self.interval_upper_quantile:
            raise ValueError("lower interval quantile must be less than upper quantile")
        return self


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_athlete_config(path: Path) -> AthleteConfig:
    return AthleteConfig.model_validate(load_yaml(path))


def load_model_config(path: Path | None) -> ModelConfig:
    if path is None:
        return ModelConfig()
    return ModelConfig.model_validate(load_yaml(path))
