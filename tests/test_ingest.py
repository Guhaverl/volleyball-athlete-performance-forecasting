from __future__ import annotations

from volley_forecast.config import AthleteConfig, AthleteSourceConfig, Settings
from volley_forecast.ingest import ingest_athlete


def test_ingest_player_page_with_fake_client(tmp_path, player_html, monkeypatch) -> None:
    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def fetch_player_profile_html(self, competition_slug, season, player_id):
            return player_html, f"https://example.test/{competition_slug}/{season}/{player_id}"

    monkeypatch.setattr("volley_forecast.ingest.VolleyballWorldClient", FakeClient)
    athlete = AthleteConfig(
        player_id="123",
        team_name="Demo United",
        sources=[AthleteSourceConfig(competition_slug="demo", season=2025)],
    )
    output = tmp_path / "processed" / "athlete.csv"
    raw_dir = tmp_path / "raw"
    frame = ingest_athlete(
        athlete,
        output_path=output,
        raw_dir=raw_dir,
        settings=Settings(vw_cache_dir=tmp_path / "cache"),
        include_dnp=True,
    )
    assert len(frame) == 3
    assert output.exists()
    assert len(list(raw_dir.glob("*.html"))) == 1
