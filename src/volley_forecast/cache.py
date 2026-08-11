"""Small file cache used to reduce repeated requests to source endpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class FileResponseCache:
    def __init__(self, directory: Path, ttl_hours: float) -> None:
        self.directory = directory
        self.ttl = timedelta(hours=ttl_hours)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if datetime.now(UTC) - fetched_at > self.ttl:
            return None
        return payload

    def set(self, key: str, *, text: str, content_type: str) -> None:
        payload = {
            "fetched_at": datetime.now(UTC).isoformat(),
            "content_type": content_type,
            "text": text,
        }
        path = self._path(key)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)
