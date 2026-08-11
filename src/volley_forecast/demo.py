"""Deterministic synthetic data for local pipeline demonstrations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from volley_forecast.schema import CANONICAL_COLUMNS


def generate_demo_frame(matches: int = 72, seed: int = 42) -> pd.DataFrame:
    if matches < 24:
        raise ValueError("Generate at least 24 matches for meaningful chronological splits")
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp("2022-05-01")]
    for _ in range(1, matches):
        dates.append(dates[-1] + pd.Timedelta(days=int(rng.integers(5, 13))))

    opponents = ["Italy", "Brazil", "Poland", "Japan", "Türkiye", "USA", "Serbia", "China"]
    rows: list[dict[str, object]] = []
    form = 0.0
    for index, match_date in enumerate(dates):
        opponent = opponents[index % len(opponents)]
        opponent_effect = {"Italy": 1.2, "Brazil": -0.8, "Poland": 0.6}.get(opponent, 0.0)
        form = 0.78 * form + float(rng.normal(0, 0.8))
        attack_points = max(0, int(round(10.5 + form + opponent_effect + rng.normal(0, 2.0))))
        block_points = max(0, int(round(1.8 + 0.25 * form + rng.normal(0, 0.9))))
        serve_points = max(0, int(round(1.4 + 0.20 * form + rng.normal(0, 0.8))))
        total_points = attack_points + block_points + serve_points
        attack_attempts = max(attack_points, int(round(attack_points / 0.43 + rng.normal(0, 2))))
        serve_attempts = max(serve_points, int(round(15 + rng.normal(0, 3))))
        reception_attempts = max(0, int(round(14 + rng.normal(0, 4))))
        team_a = "Demo United" if index % 2 == 0 else opponent
        team_b = opponent if index % 2 == 0 else "Demo United"
        rows.append(
            {
                "player_id": "demo-001",
                "player_name": "Demo Athlete",
                "position": "Outside Hitter",
                "competition_slug": "synthetic-vnl",
                "season": match_date.year,
                "match_date": match_date.date().isoformat(),
                "team_a": team_a,
                "team_b": team_b,
                "player_team": "Demo United",
                "opponent": opponent,
                "total_points": total_points,
                "attack_points": attack_points,
                "block_points": block_points,
                "serve_points": serve_points,
                "attack_errors": max(0, int(round(3 + rng.normal(0, 1.4)))),
                "attack_attempts": attack_attempts,
                "attack_success_pct": 100.0 * attack_points / max(attack_attempts, 1),
                "attack_total_actions": attack_attempts,
                "block_errors": max(0, int(round(1 + rng.normal(0, 0.7)))),
                "block_rebounds": max(0, int(round(2 + rng.normal(0, 1.0)))),
                "block_efficiency_pct": 100.0 * block_points / max(block_points + 4, 1),
                "block_total_actions": max(block_points + 4, 1),
                "serve_errors": max(0, int(round(2 + rng.normal(0, 1.0)))),
                "serve_attempts": serve_attempts,
                "serve_success_pct": 100.0 * serve_points / max(serve_attempts, 1),
                "serve_total_actions": serve_attempts,
                "reception_successful": max(0, int(round(0.58 * reception_attempts + rng.normal(0, 1)))),
                "reception_errors": max(0, int(round(1 + rng.normal(0, 0.8)))),
                "reception_attempts": reception_attempts,
                "reception_success_pct": 58.0 + float(rng.normal(0, 6)),
                "reception_total_actions": reception_attempts,
                "digs": max(0, int(round(7 + rng.normal(0, 2)))),
                "dig_errors": max(0, int(round(rng.normal(1, 0.7)))),
                "dig_receptions": max(0, int(round(10 + rng.normal(0, 3)))),
                "dig_success_pct": 64.0 + float(rng.normal(0, 7)),
                "dig_total_actions": max(1, int(round(12 + rng.normal(0, 3)))),
                "set_successful": max(0, int(round(2 + rng.normal(0, 1)))),
                "set_errors": max(0, int(round(rng.normal(0.4, 0.5)))),
                "set_attempts": max(1, int(round(3 + rng.normal(0, 1)))),
                "set_success_pct": 70.0 + float(rng.normal(0, 8)),
                "set_total_actions": max(1, int(round(3 + rng.normal(0, 1)))),
                "participated": True,
                "source_type": "synthetic_demo",
                "source_url": "generated://demo",
                "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            }
        )
    return pd.DataFrame(rows)[CANONICAL_COLUMNS]


def write_demo_csv(path: Path, *, matches: int = 72, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_frame(matches=matches, seed=seed).to_csv(path, index=False)
    return path
