# Testing

Use the Makefile targets for a consistent experience.

## Quick start

```bash
make install-dev        # Install dev deps
make test               # Unit + golden tests (no coverage)
make test-coverage      # All tests with coverage report
make test-all           # Unit + integration + workflows (non-API)
```

## Granular suites

```bash
make test-unit          # Unit tests only
make test-golden-files  # Golden file regression tests
make test-integration   # Integration tests (skips if semantic-release missing)
make test-workflows     # Workflow configuration tests
make test-api           # API tests (require GEMINI_API_KEY)
```

Notes:

- API tests require `GEMINI_API_KEY` to be set.
- Integration tests will be skipped if `semantic-release` is not installed.

## Suites overview

- characterization: golden-file behavioral tests
- workflows: end-to-end pipeline/workflow behavior
- unit: small, focused units

See Explanation → Contract-First Testing for methodology and invariants.
