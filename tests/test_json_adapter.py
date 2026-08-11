from __future__ import annotations

from volley_forecast.json_adapter import JsonMapping, get_path, parse_player_stats_json


def test_get_path_handles_dicts_and_lists() -> None:
    payload = {"data": {"items": [{"value": 9}]}}
    assert get_path(payload, "data.items.0.value") == 9
    assert get_path(payload, "data.items.3.value", "missing") == "missing"


def test_parse_configured_json() -> None:
    payload = {
        "data": {
            "player": {"name": "A. Player", "position": "Setter"},
            "matches": [
                {
                    "date": "2025-06-01",
                    "home": "Blue",
                    "away": "Red",
                    "points": {"total": "11", "attack": "8", "block": 2, "serve": 1},
                }
            ],
        }
    }
    mapping = JsonMapping(
        records_path="data.matches",
        profile={"player_name": "data.player.name", "position": "data.player.position"},
        fields={
            "match_date": "date",
            "team_a": "home",
            "team_b": "away",
            "total_points": "points.total",
            "attack_points": "points.attack",
            "block_points": "points.block",
            "serve_points": "points.serve",
        },
    )
    frame = parse_player_stats_json(
        payload,
        mapping,
        player_id="p1",
        competition_slug="vnl",
        season=2025,
        source_url="https://example.test/api",
    )
    assert len(frame) == 1
    assert frame.loc[0, "player_name"] == "A. Player"
    assert frame.loc[0, "total_points"] == 11
    assert bool(frame.loc[0, "participated"]) is True


def test_json_adapter_parses_false_string() -> None:
    payload = {"matches": [{"date": "2025-06-01", "played": "false"}]}
    mapping = JsonMapping(
        records_path="matches",
        fields={"match_date": "date", "participated": "played"},
    )
    frame = parse_player_stats_json(
        payload,
        mapping,
        player_id="p1",
        competition_slug="vnl",
        season=2025,
        source_url="https://example.test/api",
    )
    assert bool(frame.loc[0, "participated"]) is False
