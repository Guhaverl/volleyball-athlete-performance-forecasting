# Contributing

1. Create a feature branch from `main`.
2. Install `.[dev]` for core work or `.[ml,api,dev]` for the full stack.
3. Add tests for every parser, feature, or model-pipeline change.
4. Run `make format`, `make lint`, and `make test` before opening a pull request.
5. Keep live-network tests opt-in. Unit tests must use saved fixtures or mocked HTTP.

## Data-source changes

Document endpoint provenance, expected response shape, caching behavior, and a
fallback. Never add credentials, session cookies, bypasses, or code that attempts
unauthorized access. Changes that increase request frequency must explain why the
new rate is necessary.

## Modelling changes

Report chronological validation metrics and compare against both last-value and
rolling-mean baselines. A lower training loss alone is not evidence of improvement.
