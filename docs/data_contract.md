# Canonical athlete-match data contract

Each row represents one athlete in one match. Dates use ISO `YYYY-MM-DD` and
training sorts them chronologically.

## Identity and context

| Column | Type | Meaning |
|---|---|---|
| `player_id` | string | Stable source identifier |
| `player_name` | string | Display name |
| `position` | string | Playing role when available |
| `competition_slug` | string | Source competition path slug |
| `season` | integer | Competition season |
| `match_date` | date | Match date |
| `team_a`, `team_b` | string | Match participants |
| `player_team` | string | Athlete's team |
| `opponent` | string | Opposing team |
| `participated` | boolean | Whether activity or points indicate participation |

## Forecast targets

`total_points`, `attack_points`, `block_points`, and `serve_points`.

## Supporting statistics

Attack, block, serve, reception, dig, and set actions are represented by the
columns declared in `src/volley_forecast/schema.py`. Missing values remain null
at ingestion and are imputed to zero only inside sequence construction. That
keeps the raw canonical table auditable.

## Provenance

| Column | Meaning |
|---|---|
| `source_type` | `authorized_json`, `public_player_page`, or `synthetic_demo` |
| `source_url` | Exact source request or page |
| `retrieved_at` | UTC retrieval timestamp |

## Data-quality rules

- Invalid dates fail validation.
- Exact match keys are deduplicated, keeping the latest row.
- DNP rows are retained during collection but excluded from training by default.
- A configurable minimum number of participating matches is required.
- Raw responses should not be committed when licenses or terms prohibit it.
