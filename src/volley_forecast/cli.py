"""Command-line interface for ingestion, training, and forecasting."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from volley_forecast.client import VolleyballWorldClient
from volley_forecast.config import Settings, load_athlete_config, load_model_config
from volley_forecast.demo import write_demo_csv
from volley_forecast.ingest import ingest_athlete
from volley_forecast.logging_config import configure_logging

app = typer.Typer(no_args_is_help=True, help="Volleyball athlete stat forecasting toolkit.")
console = Console()


def _parse_iso_date(value: str, option_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must use YYYY-MM-DD") from exc


def _write_json(payload: Any, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if output is None:
        console.print_json(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(f"Wrote {output}")


@app.command("discover-competitions")
def discover_competitions(
    year: int = typer.Option(..., min=2000, max=2100),
    month: int = typer.Option(..., min=1, max=12),
    output: Path | None = typer.Option(None, help="Optional raw JSON output path."),
) -> None:
    """List competitions returned by Volleyball World's global schedule route."""
    configure_logging(Settings().log_level)
    with VolleyballWorldClient(Settings()) as client:
        payload = client.fetch_competitions(year, month)
    if output is not None:
        _write_json(payload, output)
        return
    competitions = payload.get("competitions", []) if isinstance(payload, dict) else []
    table = Table(title=f"Volleyball World competitions: {year}-{month:02d}")
    for column in ("Name", "Discipline", "Start", "End", "Destination"):
        table.add_column(column)
    for item in competitions:
        if not isinstance(item, dict):
            continue
        table.add_row(
            str(item.get("competitionFullName") or item.get("name") or ""),
            str(item.get("discipline") or ""),
            str(item.get("startDate") or ""),
            str(item.get("endDate") or ""),
            str(item.get("destination") or ""),
        )
    console.print(table)


@app.command("fetch-tournament")
def fetch_tournament(
    start: str = typer.Option(..., help="Inclusive YYYY-MM-DD date."),
    end: str = typer.Option(..., help="Inclusive YYYY-MM-DD date."),
    tournament_no: int = typer.Option(..., min=1),
    output: Path = typer.Option(...),
) -> None:
    """Save a tournament-level API response for source inspection."""
    configure_logging(Settings().log_level)
    with VolleyballWorldClient(Settings()) as client:
        payload = client.fetch_tournament(
            _parse_iso_date(start, "--start"),
            _parse_iso_date(end, "--end"),
            tournament_no,
        )
    _write_json(payload, output)


@app.command("ingest-player")
def ingest_player_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("data/processed/athlete_matches.csv")),
    raw_dir: Path = typer.Option(Path("data/raw")),
    include_dnp: bool = typer.Option(True, help="Retain non-participation rows in saved data."),
    continue_on_error: bool = typer.Option(False),
) -> None:
    """Ingest configured seasons into the canonical athlete-match table."""
    settings = Settings()
    configure_logging(settings.log_level)
    athlete = load_athlete_config(config)
    frame = ingest_athlete(
        athlete,
        output_path=output,
        raw_dir=raw_dir,
        settings=settings,
        include_dnp=include_dnp,
        continue_on_error=continue_on_error,
    )
    console.print(f"Wrote {len(frame)} rows to {output}")


@app.command("generate-demo-data")
def generate_demo_data_command(
    output: Path = typer.Option(Path("data/sample/demo_athlete_matches.csv")),
    matches: int = typer.Option(72, min=24),
    seed: int = typer.Option(42),
) -> None:
    """Generate deterministic synthetic data; it is never presented as real sport data."""
    write_demo_csv(output, matches=matches, seed=seed)
    console.print(f"Wrote {matches} synthetic matches to {output}")


@app.command("analyze")
def analyze_command(
    data: Path = typer.Option(..., exists=True, dir_okay=False),
    recent_window: int = typer.Option(5, min=2),
    output: Path | None = typer.Option(None),
) -> None:
    """Summarize athlete form, trends, opponent splits, and rest effects."""
    from volley_forecast.analysis import build_athlete_analysis

    result = build_athlete_analysis(pd.read_csv(data), recent_window=recent_window)
    _write_json(result, output)


@app.command("train")
def train_command(
    data: Path = typer.Option(..., exists=True, dir_okay=False),
    model_dir: Path = typer.Option(Path("artifacts/model")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Train, evaluate, and save a TensorFlow athlete model."""
    from volley_forecast.training import train_model

    settings = Settings()
    configure_logging(settings.log_level)
    result = train_model(data_path=data, model_dir=model_dir, config=load_model_config(config))
    promotion = result["metadata"]["promotion"]
    console.print(
        f"Saved model to {model_dir}. Status: {promotion['status']}; "
        f"{promotion['metric']}={promotion['candidate_score']:.3f}."
    )


@app.command("forecast")
def forecast_command(
    model_dir: Path = typer.Option(..., exists=True, file_okay=False),
    history: Path = typer.Option(..., exists=True, dir_okay=False),
    next_date: str = typer.Option(..., help="Forecast date in YYYY-MM-DD format."),
    opponent: str | None = typer.Option(None),
    mc_samples: int = typer.Option(100, min=10, max=1000),
    output: Path | None = typer.Option(None),
) -> None:
    """Forecast the configured targets for one athlete's next match."""
    from volley_forecast.inference import ForecastEngine

    engine = ForecastEngine(model_dir)
    result = engine.forecast(
        pd.read_csv(history),
        next_date=_parse_iso_date(next_date, "--next-date"),
        opponent=opponent,
        mc_samples=mc_samples,
    )
    _write_json(result, output)


@app.command("serve")
def serve_command(
    model_dir: Path = typer.Option(..., exists=True, file_okay=False),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    """Run the optional FastAPI inference service."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter('API dependencies are missing; install with ".[api]"') from exc
    from volley_forecast.service import create_app

    uvicorn.run(create_app(model_dir), host=host, port=port)


if __name__ == "__main__":
    app()
