# API Reference Overview

Use this page as a map to the public API and the most relevant internals.
If you’re new, first skim the Quickstart, then return here to jump into details.

- Quickstart: [tutorials/quickstart.md](../tutorials/quickstart.md)

Stability and versioning

- The core public API is approaching v1.0. From 1.0 onward, breaking changes will follow semantic versioning and include a deprecation/transition period.
- The documented internals are provided for transparency but are not part of the stability contract.
- Decoupled modules under `extensions/` and the upcoming `research/` area are pre-1.0 and may change without deprecation until their first stable release.

Core API

- Executor: [api/executor.md](api/executor.md)
- Frontdoor: [api/frontdoor.md](api/frontdoor.md)
- Types: [api/types.md](api/types.md)
- Config: [api/config.md](api/config.md)
- Telemetry: [api/telemetry.md](api/telemetry.md)
- Core Types: [api/core.md](api/core.md)

Internals

- Pipeline: [internals/pipeline.md](internals/pipeline.md)
- Prompts: [internals/prompts.md](internals/prompts.md)
- Adapters: [internals/adapters.md](internals/adapters.md)
- Results: [internals/results.md](internals/results.md)

Notes

- Pages above are powered by mkdocstrings and reflect current docstrings under `src/`.
- Internal note: docstring audit and conventions are tracked in `dev/internal_only/audits/docstring_audit.md`.
