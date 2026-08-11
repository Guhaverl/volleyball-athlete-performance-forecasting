from __future__ import annotations

from volley_forecast.parser import parse_player_profile_html


def test_parse_player_profile_tables(player_html: str) -> None:
    frame = parse_player_profile_html(
        player_html,
        player_id="123",
        competition_slug="demo-competition",
        season=2025,
        source_url="https://example.test/player/123",
    )

    assert len(frame) == 3
    assert frame.loc[0, "player_name"] == "Demo Athlete"
    assert frame.loc[0, "position"] == "Outside Hitter"
    assert frame.loc[0, "player_team"] == "Demo United"
    assert frame.loc[0, "opponent"] == "Italy"
    assert frame.loc[1, "player_team"] == "Demo United"
    assert frame.loc[1, "opponent"] == "Brazil"
    assert frame.loc[0, "total_points"] == 15
    assert frame.loc[1, "attack_attempts"] == 31
    assert frame.loc[1, "serve_success_pct"] == 11.1
    assert bool(frame.loc[0, "participated"]) is True
    assert bool(frame.loc[2, "participated"]) is False
