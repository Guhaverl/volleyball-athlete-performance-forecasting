"""Optional FastAPI service for a saved forecast model."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from volley_forecast.exceptions import VolleyForecastError
from volley_forecast.inference import ForecastEngine


class ForecastRequest(BaseModel):
    history: list[dict[str, Any]] = Field(min_length=1)
    next_date: date
    opponent: str | None = None
    mc_samples: int = Field(default=100, ge=10, le=1000)


def create_app(model_dir: Path) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError(
            'API dependencies are missing. Install with: pip install -e ".[api]"'
        ) from exc

    engine = ForecastEngine(model_dir)
    app = FastAPI(
        title="Volleyball Athlete Forecast API",
        version="0.1.0",
        description="Athlete-specific next-match forecasts from a saved TensorFlow model.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_dir": str(model_dir),
            "athlete": engine.metadata.get("athlete", {}),
        }

    @app.post("/forecast")
    def forecast(request: ForecastRequest) -> dict[str, Any]:
        try:
            return engine.forecast(
                pd.DataFrame(request.history),
                next_date=request.next_date,
                opponent=request.opponent,
                mc_samples=request.mc_samples,
            )
        except (ValueError, FileNotFoundError, VolleyForecastError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
