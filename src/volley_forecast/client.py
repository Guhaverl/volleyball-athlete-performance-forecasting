"""HTTP client for public or explicitly authorized Volleyball World routes."""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from volley_forecast.cache import FileResponseCache
from volley_forecast.config import Settings
from volley_forecast.exceptions import SourceAdapterError

LOGGER = logging.getLogger(__name__)


class VolleyballWorldClient:
    """Rate-limited, cached client with all source URLs isolated in one module."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: FileResponseCache | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.cache = cache or FileResponseCache(
            self.settings.vw_cache_dir, self.settings.vw_cache_ttl_hours
        )
        self._last_request_at = 0.0
        self._client = httpx.Client(
            timeout=self.settings.vw_timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": self.settings.vw_user_agent,
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            },
            transport=transport,
        )

    def __enter__(self) -> "VolleyballWorldClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _absolute_url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return urljoin(f"{self.settings.vw_base_url}/", path_or_url.lstrip("/"))

    def _respect_rate_limit(self) -> None:
        interval = max(self.settings.vw_request_interval_seconds, 0.0)
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def get_text(self, path_or_url: str, *, use_cache: bool = True) -> str:
        url = self._absolute_url(path_or_url)
        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                return str(cached["text"])

        self._respect_rate_limit()
        LOGGER.info("GET %s", url)
        response = self._client.get(url)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        text = response.text
        if use_cache:
            self.cache.set(
                url,
                text=text,
                content_type=response.headers.get("content-type", ""),
            )
        return text

    def get_json(self, path_or_url: str, *, use_cache: bool = True) -> dict[str, Any]:
        text = self.get_text(path_or_url, use_cache=use_cache)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceAdapterError(f"Expected JSON from {path_or_url}") from exc
        if not isinstance(payload, dict):
            raise SourceAdapterError(f"Expected a JSON object from {path_or_url}")
        return payload

    def fetch_competitions(self, year: int, month: int | None = None) -> dict[str, Any]:
        if month is None:
            path = f"/api/v1/globalschedule/competitions/{year}/"
        else:
            if month < 1 or month > 12:
                raise ValueError("month must be between 1 and 12")
            path = f"/api/v1/globalschedule/competitions/{year}/{month}"
        return self.get_json(path)

    def fetch_global_schedule(self, start_date: date, end_date: date) -> dict[str, Any]:
        if end_date < start_date:
            raise ValueError("end_date cannot precede start_date")
        return self.get_json(f"/api/v1/globalschedule/{start_date.isoformat()}/{end_date.isoformat()}")

    def fetch_tournament(
        self, start_date: date, end_date: date, tournament_no: int
    ) -> dict[str, Any]:
        if end_date < start_date:
            raise ValueError("end_date cannot precede start_date")
        path = (
            f"/api/v1/volley-tournament/{start_date.isoformat()}/"
            f"{end_date.isoformat()}/{tournament_no}"
        )
        return self.get_json(path)

    def player_profile_url(self, competition_slug: str, season: int, player_id: str) -> str:
        path = f"/volleyball/competitions/{competition_slug}/{season}/players/{player_id}"
        return self._absolute_url(path)

    def fetch_player_profile_html(
        self, competition_slug: str, season: int, player_id: str
    ) -> tuple[str, str]:
        url = self.player_profile_url(competition_slug, season, player_id)
        return self.get_text(url), url

    def fetch_configured_player_json(
        self, competition_slug: str, season: int, player_id: str
    ) -> tuple[dict[str, Any], str]:
        template = self.settings.vw_player_stats_url_template
        if not template:
            raise SourceAdapterError("VW_PLAYER_STATS_URL_TEMPLATE is not configured")
        url = template.format(
            base_url=self.settings.vw_base_url,
            competition_slug=competition_slug,
            season=season,
            player_id=player_id,
        )
        absolute = self._absolute_url(url)
        return self.get_json(absolute), absolute

    def save_json(self, payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
