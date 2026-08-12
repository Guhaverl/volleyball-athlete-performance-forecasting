from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from volley_forecast.cache import FileResponseCache
from volley_forecast.client import VolleyballWorldClient
from volley_forecast.config import Settings


def test_client_builds_routes_and_uses_cache(tmp_path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=json.dumps({"competitions": [{"name": "Demo"}]}))

    settings = Settings(
        vw_base_url="https://example.test",
        vw_request_interval_seconds=0,
        vw_cache_dir=tmp_path / "cache",
    )
    cache = FileResponseCache(tmp_path / "cache", ttl_hours=24)
    with VolleyballWorldClient(
        settings,
        transport=httpx.MockTransport(handler),
        cache=cache,
    ) as client:
        first = client.fetch_competitions(2025, 6)
        second = client.fetch_competitions(2025, 6)
        url = client.player_profile_url("league", 2025, "77")

    assert first == second
    assert calls == ["/api/v1/globalschedule/competitions/2025/6"]
    assert url == "https://example.test/volleyball/competitions/league/2025/players/77"


def test_client_rejects_bad_ranges(tmp_path) -> None:
    settings = Settings(vw_cache_dir=tmp_path, vw_request_interval_seconds=0)
    with VolleyballWorldClient(
        settings, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    ) as client:
        with pytest.raises(ValueError, match="precede"):
            client.fetch_global_schedule(date(2025, 2, 1), date(2025, 1, 1))
        with pytest.raises(ValueError, match="month"):
            client.fetch_competitions(2025, 13)
