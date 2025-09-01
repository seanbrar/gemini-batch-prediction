# Legacy Characterization Goldens (Historical Artifacts)

These YAML files are historical golden artifacts preserved for reference. They reflect the behavior of the pre‑v1 architecture and are intentionally not executed as tests.

Notes

- These files do not represent the current v1 API surface or intended behavior.
- The former legacy tests have been removed to reduce maintenance and avoid confusion.
- When a new characterization suite is built for the v1 API, it will live under `tests/characterization/` and may use `pytest-golden` (kept as a dev dependency).

Scope

- This directory is not part of any test lane. CI and local `pytest` runs do not execute anything here.
