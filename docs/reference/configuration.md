# Configuration — Reference

> Audience: users who need precise rules, APIs, and environment details.

!!! info "See also"
    - How‑To: [Configuration](../how-to/configuration.md)
    - ADR‑0007: [Configuration Resolution & Immutability](../explanation/decisions/ADR-0007-configuration.md)

---

## Overview

- Precedence: Programmatic > Environment > Project file > Home file > Defaults.
- Files: project `pyproject.toml` and optional home `~/.config/gemini_batch.toml`.
- Profiles: select a table under `[tool.gemini_batch.profiles.<name>]`.
- Immutable runtime: `resolve_config()` returns a `FrozenConfig` (handlers should not re‑resolve).

---

## Precedence & Profiles

- Order per field: defaults → home → project → env → overrides (last‑wins).
- Profiles: choose a profile section from TOML files via:
  - Env: set `GEMINI_BATCH_PROFILE` (e.g., `dev`, `prod`).
  - Code: `resolve_config(profile="dev")`.
- The home file does not override values set in the project file for the same field unless the project omits that field.

---

## File Discovery & Overrides

- Project file: `pyproject.toml` at the working directory by default.
  - Override path with `GEMINI_BATCH_PYPROJECT_PATH` (absolute or relative).
- Home file: `~/.config/gemini_batch.toml`.
  - Override file path with `GEMINI_BATCH_CONFIG_HOME` (absolute or relative path to the TOML file).
- Fallbacks: If `Path.home()` is not available (restricted environments), the loader may use a cwd‑based fallback for the home file.
- Structure:

```toml
[tool.gemini_batch]
model = "gemini-2.0-flash"
use_real_api = false

[tool.gemini_batch.profiles.dev]
model = "gemini-2.0-flash"

[tool.gemini_batch.profiles.prod]
model = "gemini-2.0-pro"
use_real_api = true
```

---

## Provider Inference

Provider is inferred from the `model` using priority rules:

1) Exact model names (highest priority)
2) Versioned patterns (e.g., `gemini-<major>.<minor>`)
3) Simple prefixes (fallback)

Built‑in mappings include:

- `gemini-*` → `google`
- `gpt-*` → `openai`
- `claude-*` → `anthropic`

Example:

```python
from gemini_batch.config import resolve_config

cfg = resolve_config(overrides={"model": "gpt-4o"})
assert cfg.provider == "openai"
```

---

## Extra Fields Validation

Unknown keys are preserved in `FrozenConfig.extra` and validated with friendly warnings according to naming patterns:

- `*_timeout`: expect `int` seconds
- `*_url`: expect `str` URL
- `experimental_*`: accepted; API may change
- `legacy_*`: accepted; emits a deprecation warning

Warnings are emitted via `warnings.warn` and do not fail resolution.

---

## Diagnostics & Controls

- Redaction: `api_key` and other sensitive values are redacted in string representations and audits.
- Debug audit: enable with `GEMINI_BATCH_DEBUG_CONFIG=1|true|yes|on`; emits a redacted audit via Python warnings when resolving without `explain=True`.
- Environment snapshot: `check_environment()` returns current `GEMINI_*` and `GEMINI_BATCH_*` variables with sensitive values redacted.
- Doctor: `doctor()` returns a list of diagnostic messages and advisories (e.g., missing key when `use_real_api=True`).

CLI equivalents:

- `gb-config show` — effective redacted config (JSON)
- `gb-config audit` — field origins and layer summary
- `gb-config doctor` — actionable checks and advisories
- `gb-config env` — redacted environment snapshot

---

## Public API

- `resolve_config(overrides: Mapping | None = None, profile: str | None = None, *, explain: bool = False)` → `FrozenConfig` or `(FrozenConfig, SourceMap)` when `explain=True`.
  - Returns an immutable `FrozenConfig` with fields: `model`, `api_key`, `use_real_api`, `enable_caching`, `ttl_seconds`, `telemetry_enabled`, `tier`, `request_concurrency`, `provider`, and `extra` (dict of unknown keys).
  - When `explain=True`, also returns a `SourceMap` describing where each field originated.

```python
from gemini_batch.config import resolve_config

cfg, src = resolve_config(explain=True)
print(cfg.provider)  # e.g., "google"
print(src["model"].origin)  # one of: default|home|project|env|overrides
print(src["api_key"].env_key)  # e.g., "GEMINI_API_KEY" (redacted in logs)
```

- `config_scope(cfg_or_overrides: FrozenConfig | Mapping | None = None, *, profile: str | None = None, **overrides)`
  - Context manager to run within a temporary configuration. If given a mapping, it resolves to a `FrozenConfig` first.
  - Scope only affects resolution time; handlers operate on the resolved snapshot.

- `check_environment()` → `dict[str, str]`
  - Returns current `GEMINI_BATCH_*` variables with sensitive values redacted.

- `doctor()` → `list[str]`
  - Simple health check with actionable messages.

---

## Environment Variables

- `GEMINI_API_KEY`: API key (required when `use_real_api=true`).
- `GEMINI_BATCH_MODEL`: Optional model override; equivalent to setting `model`.
- `GEMINI_BATCH_USE_REAL_API`: `true`/`false`; toggles real API access.
- `GEMINI_BATCH_TIER`: Billing tier to tune rate limits; one of `free`, `tier_1`, `tier_2`, `tier_3`.
- `GEMINI_BATCH_ENABLE_CACHING`: `true`/`false`; enables context caching features.
- `GEMINI_BATCH_TTL_SECONDS`: Integer TTL for cache entries (seconds; default 3600).
- `GEMINI_BATCH_PROFILE`: Selects a TOML profile (e.g., `dev`, `prod`).
- `GEMINI_BATCH_PYPROJECT_PATH`: Override path to project `pyproject.toml`.
- `GEMINI_BATCH_CONFIG_HOME`: Override path to home TOML (`~/.config/gemini_batch.toml`).
- `GEMINI_BATCH_DEBUG_CONFIG`: Enable redacted debug audit emission.
- `GEMINI_BATCH_REQUEST_CONCURRENCY`: Client-side fan-out bound for vectorized requests (int, default: 6).

Tip: Use a `.env` file during development; the resolver attempts to load it if `python-dotenv` is installed.
