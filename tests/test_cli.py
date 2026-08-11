from __future__ import annotations

from typer.testing import CliRunner

from volley_forecast.cli import app


def test_generate_demo_cli(tmp_path) -> None:
    output = tmp_path / "demo.csv"
    result = CliRunner().invoke(
        app,
        ["generate-demo-data", "--output", str(output), "--matches", "24", "--seed", "9"],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
