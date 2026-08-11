# Model card template

Training writes a completed card into each artifact directory. Review it before
serving or publishing predictions.

## Intended use

Exploratory next-match forecasts for one athlete using prior match statistics and
known next-match context. Suitable for coaching research, workload discussions,
and reproducible sports-analytics education when source-data rights permit.

## Out-of-scope use

Do not use predictions as medical advice, automated team-selection decisions,
contract or employment decisions, or a guarantee for wagering outcomes.

## Evaluation standard

Report chronological held-out MAE, RMSE, and sMAPE per target. Also report
last-value and rolling-mean baselines. Mark the TensorFlow model
`baseline_preferred` whenever it does not beat the strongest baseline on the
configured promotion metric.

## Known limitations

Player samples are small. Playing time, injuries, tactical role, opponent level,
travel, roster decisions, and rule changes may be missing. Monte Carlo dropout
bands measure only one form of model uncertainty and are not calibrated
probability guarantees.
