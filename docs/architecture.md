# Architecture

## Design goals

1. Keep every Volleyball World URL and request policy in one client.
2. Convert every source into one canonical athlete-match table.
3. Prevent future information from entering training features.
4. Compare TensorFlow with transparent persistence baselines.
5. Save everything required to reproduce inference.

## Components

```text
VolleyballWorldClient
  ├── global competition/schedule context
  ├── tournament payload context
  ├── configured authorized athlete JSON
  └── public player-page HTML fallback
           │
           ▼
parser/json_adapter -> canonical CSV -> validation -> feature engineering
                                                    │
                                                    ▼
                          chronological windows -> baseline + LSTM evaluation
                                                    │
                                                    ▼
                           model.keras + scalers + metadata + metrics + card
                                                    │
                                                    ▼
                                           CLI / FastAPI inference
```

## Modelling boundary

A model bundle belongs to one athlete and one coherent role. It is intentionally
not a league-wide player-ranking model. Train a separate bundle when position,
competition level, scoring rules, or athlete role changes materially.

## Leakage controls

- Rows are sorted by match date.
- Training, validation, and test sets are contiguous in time.
- Scalers are fit on training samples only.
- Opponent history uses shifted values, never the current label.
- Keras training keeps `shuffle=False`.
- Test data is evaluated only after model fitting.
