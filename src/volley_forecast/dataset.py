"""Chronological sequence construction and splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceBundle:
    x_history: np.ndarray
    x_context: np.ndarray
    y: np.ndarray
    label_positions: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class SequenceSplit:
    train: SequenceBundle
    val: SequenceBundle
    test: SequenceBundle


def make_sequences(
    frame: pd.DataFrame,
    *,
    history_steps: int,
    history_features: list[str],
    context_features: list[str],
    targets: list[str],
) -> SequenceBundle:
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    required = [*history_features, *context_features, *targets]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sequence columns: {', '.join(missing)}")
    if len(frame) <= history_steps:
        raise ValueError("Not enough rows to build a single sequence")

    history_matrix = frame[history_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    context_matrix = frame[context_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_matrix = frame[targets].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    histories: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positions: list[int] = []
    for label_position in range(history_steps, len(frame)):
        histories.append(
            history_matrix.iloc[label_position - history_steps : label_position].to_numpy(
                dtype=np.float32
            )
        )
        contexts.append(context_matrix.iloc[label_position].to_numpy(dtype=np.float32))
        labels.append(target_matrix.iloc[label_position].to_numpy(dtype=np.float32))
        positions.append(label_position)

    return SequenceBundle(
        x_history=np.stack(histories),
        x_context=np.stack(contexts),
        y=np.stack(labels),
        label_positions=np.asarray(positions, dtype=np.int64),
    )


def _slice(bundle: SequenceBundle, start: int, stop: int) -> SequenceBundle:
    return SequenceBundle(
        x_history=bundle.x_history[start:stop],
        x_context=bundle.x_context[start:stop],
        y=bundle.y[start:stop],
        label_positions=bundle.label_positions[start:stop],
    )


def chronological_split(
    bundle: SequenceBundle,
    *,
    train_fraction: float,
    val_fraction: float,
) -> SequenceSplit:
    total = len(bundle)
    if total < 6:
        raise ValueError("At least six sequences are required for train/validation/test splitting")
    train_stop = max(2, int(total * train_fraction))
    val_size = max(1, int(total * val_fraction))
    val_stop = train_stop + val_size
    if val_stop >= total:
        val_stop = total - 1
    if train_stop >= val_stop:
        train_stop = val_stop - 1
    return SequenceSplit(
        train=_slice(bundle, 0, train_stop),
        val=_slice(bundle, train_stop, val_stop),
        test=_slice(bundle, val_stop, total),
    )
