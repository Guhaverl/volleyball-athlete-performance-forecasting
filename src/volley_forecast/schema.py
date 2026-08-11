"""Canonical athlete-match schema shared by ingestion and modelling."""

from __future__ import annotations

TARGET_COLUMNS = [
    "total_points",
    "attack_points",
    "block_points",
    "serve_points",
]

MATCH_KEY_COLUMNS = ["player_id", "competition_slug", "season", "match_date", "team_a", "team_b"]

NUMERIC_STAT_COLUMNS = [
    *TARGET_COLUMNS,
    "attack_errors",
    "attack_attempts",
    "attack_success_pct",
    "attack_total_actions",
    "block_errors",
    "block_rebounds",
    "block_efficiency_pct",
    "block_total_actions",
    "serve_errors",
    "serve_attempts",
    "serve_success_pct",
    "serve_total_actions",
    "reception_successful",
    "reception_errors",
    "reception_attempts",
    "reception_success_pct",
    "reception_total_actions",
    "digs",
    "dig_errors",
    "dig_receptions",
    "dig_success_pct",
    "dig_total_actions",
    "set_successful",
    "set_errors",
    "set_attempts",
    "set_success_pct",
    "set_total_actions",
]

CANONICAL_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "competition_slug",
    "season",
    "match_date",
    "team_a",
    "team_b",
    "player_team",
    "opponent",
    *NUMERIC_STAT_COLUMNS,
    "participated",
    "source_type",
    "source_url",
    "retrieved_at",
]

DEFAULT_HISTORY_FEATURES = [
    *TARGET_COLUMNS,
    "attack_errors",
    "attack_attempts",
    "attack_success_pct",
    "block_errors",
    "block_rebounds",
    "serve_errors",
    "serve_attempts",
    "serve_success_pct",
    "reception_successful",
    "reception_errors",
    "reception_attempts",
    "digs",
    "set_successful",
    "days_rest",
]

DEFAULT_CONTEXT_FEATURES = [
    "days_rest",
    "season_day_sin",
    "season_day_cos",
    "opponent_prior_total_points_mean",
]
