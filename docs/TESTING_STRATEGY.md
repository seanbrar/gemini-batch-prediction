# Testing Strategy

This document outlines the optimized testing cadence implemented in the CI/CD pipeline to balance fast feedback with thorough validation.

## Testing Cadence Overview

| Event | Test Suite | Target Time | What's Tested |
|-------|------------|-------------|---------------|
| Feature Push | `test-fast` | < 30s | Contracts, unit, characterization |
| Draft PR | `test-fast` | < 30s | Same with DEBUG logging |
| Ready PR | `test-pr` | < 2 min | Progressive + integration + workflows |
| Main Merge | `test-main` | < 5 min | All tests incl. slow + coverage |
| Release | `test-main` + API | < 10 min | Enable API via env |

Note: API tests are skipped unless both `GEMINI_API_KEY` and `ENABLE_API_TESTS=1` are set.

## Makefile Targets

### Fast Test Suites

```bash
# Fast tests only (~30s): contracts + unit + characterization
make test-fast

# Fast tests with timing information
make test-fast-timed
```

### Default Quick Suite

```bash
# Unit + golden regression (no coverage)
make test
```

### Progressive Testing

```bash
# Progressive tests with fail-fast (contracts → unit → characterization)
make test-progressive
```

### CI-Optimized Suites

```bash
# Pull Request suite (no slow tests)
make test-pr

# Main branch suite (everything + coverage)
make test-main
```

### Coverage

```bash
# Run all tests with coverage (HTML at coverage_html_report/)
make test-coverage

# Adjust coverage threshold (default 40)
COVERAGE_FAIL_UNDER=60 make test-coverage
```

### Individual Test Categories

```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Workflow configuration tests
make test-workflows

# Contract tests only
make test-contracts

# Slow tests only
make test-slow

# API tests (requires GEMINI_API_KEY)
make test-api

# Golden file regression tests
make test-golden-files
```

### Logging Verbosity

```bash
# Increase pytest log verbosity for any target
TEST_LOG_LEVEL=DEBUG make test-fast
```

## CI Workflow Logic

The CI workflow automatically selects the appropriate test suite based on the event:

### Feature Branch Pushes

- **Goal:** Fast feedback during development
- **Command:** `make test-fast`
- **Benefits:** ~30 seconds feedback, catches most issues early

### Draft Pull Requests

- **Goal:** Development feedback with debugging support
- **Command:** `make test-fast` (with DEBUG logging)
- **Benefits:** Same fast tests with enhanced logging for troubleshooting

### Ready Pull Requests

- **Goal:** Comprehensive validation before merge
- **Command:** `make test-pr` (progressive + integration + workflows)
- **Benefits:** Progressive tests fail fast on architectural violations

### Main Branch

- **Goal:** Full validation and metrics
- **Command:** `make test-main` (everything + coverage)
- **Benefits:** Complete validation with coverage reporting

## Progressive Testing Strategy

The `test-progressive` target runs tests in order of importance with fail-fast behavior:

1. **Architectural Contracts** - Fast guards for architectural violations
2. **Unit Tests** - Core functionality validation
3. **Characterization Tests** - Golden file regression tests

If any step fails, the entire suite stops immediately, providing fast feedback on the most critical issues.

## Benefits

1. **Fast Developer Feedback**: Feature branches get results in ~30 seconds
2. **Progressive Validation**: Architectural violations fail immediately
3. **Resource Efficiency**: Slow tests only run when necessary
4. **Clear Escalation**: More thorough testing as code approaches production
5. **Debugging Support**: Draft PRs automatically get DEBUG logging
6. **Flexibility**: Easy to adjust what runs where by updating Makefile targets

## Customization

To modify the testing strategy:

1. **Add new test categories**: Update the Makefile with new targets
2. **Adjust timing**: Modify the pytest markers (`slow`, `api`, etc.)
3. **Change CI logic**: Update the conditional expressions in `ci.yml`
4. **Add matrix testing**: Enhance `reusable-checks.yml` with matrix support

## Debugging

For debugging test failures:

```bash
# Run with DEBUG logging
TEST_LOG_LEVEL=DEBUG make test-fast

# Run with timing information
make test-fast-timed

# Run specific test categories
make test-unit
make test-contracts
```

## Performance Monitoring

Use the timing targets to monitor test performance:

```bash
# Check fast test timing
make test-fast-timed

# Monitor specific test categories
time make test-unit
time make test-contracts
```
