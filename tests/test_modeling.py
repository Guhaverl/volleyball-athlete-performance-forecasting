from __future__ import annotations

import numpy as np
import pytest

from volley_forecast.modeling import monte_carlo_predictions


class CounterModel:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, inputs, training=False):
        self.value += 1
        return np.full((inputs["history"].shape[0], 2), self.value, dtype=float)


def test_monte_carlo_predictions() -> None:
    result = monte_carlo_predictions(
        CounterModel(),
        x_history=np.zeros((1, 3, 2), dtype=float),
        x_context=np.zeros((1, 2), dtype=float),
        samples=4,
    )
    assert result.shape == (4, 1, 2)
    assert result[-1, 0, 0] == 4
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_predictions(
            CounterModel(),
            x_history=np.zeros((1, 3, 2)),
            x_context=np.zeros((1, 2)),
            samples=0,
        )
