from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volley_forecast.demo import generate_demo_frame


@pytest.fixture
def demo_frame() -> pd.DataFrame:
    return generate_demo_frame(matches=36, seed=7)


@pytest.fixture
def player_html() -> str:
    path = Path(__file__).parent / "fixtures" / "player_page.html"
    return path.read_text(encoding="utf-8")
