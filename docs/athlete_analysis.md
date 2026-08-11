# Athlete-specific analysis

Run descriptive analysis before training:

```bash
volley-forecast analyze \
  --data data/processed/athlete_matches.csv \
  --recent-window 5 \
  --output artifacts/athlete/analysis.json
```

The report includes:

- full-history and recent means for each target;
- recent-versus-career deltas and short-term linear slopes;
- opponent-specific match counts and total-point summaries;
- total-point averages grouped by rest-day bands;
- sample dates and athlete identity.

These summaries are descriptive, not causal. An apparent rest or opponent effect
may instead reflect playing time, competition strength, role, injury, or other
unobserved variables.
