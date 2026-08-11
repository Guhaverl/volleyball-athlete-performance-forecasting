from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import volley_forecast.service as service


class FakeEngine:
    def __init__(self, model_dir: Path) -> None:
        self.metadata = {"athlete": {"player_id": "demo"}}

    def forecast(self, history, **kwargs):
        return {"rows": len(history), "date": kwargs["next_date"].isoformat()}


def test_service_accepts_json_body(monkeypatch) -> None:
    monkeypatch.setattr(service, "ForecastEngine", FakeEngine)
    client = TestClient(service.create_app(Path("unused")))
    response = client.post(
        "/forecast",
        json={"history": [{"total_points": 10}], "next_date": "2026-08-20", "mc_samples": 10},
    )
    assert response.status_code == 200
    assert response.json() == {"rows": 1, "date": "2026-08-20"}
