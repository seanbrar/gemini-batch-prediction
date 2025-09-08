# Configuration — How‑To

Last reviewed: 2025-09

> Audience: practitioners configuring or testing now. Task‑focused.

!!! info "See also"
    - Reference: [Configuration](../reference/configuration.md) — precedence, file discovery, provider inference rules, API surface.
    - ADR‑0007: [Configuration Resolution & Immutability](../explanation/decisions/ADR-0007-configuration.md)

---

## 1) Quick Start: Use environment variables only

Goal: Run with the real API by setting just two variables.

```bash
export GEMINI_API_KEY="<your key>"
export GEMINI_BATCH_MODEL="gemini-2.0-flash"
export GEMINI_BATCH_USE_REAL_API="true"
```

```python
from gemini_batch.config import resolve_config
config = resolve_config()  # Direct FrozenConfig - clean and simple
print(f"Using {config.model} with provider {config.provider}")
```

Notes:

- Provider is auto‑inferred (rules in Reference → Configuration → Provider inference).
- If `use_real_api=True`, an `api_key` is required; otherwise mock mode is allowed.
- Secrets are redacted in string representations and audits.

---

## 2) Project defaults in `pyproject.toml`

Goal: Share defaults for your team in‑repo.

```toml
# pyproject.toml
[tool.gemini_batch]
model = "gemini-2.0-flash"
enable_caching = false
use_real_api = false
ttl_seconds = 3600

# Custom fields with conventional patterns
request_timeout = 30  # *_timeout pattern
api_url = "https://api.example.com"  # *_url pattern
```

```python
from gemini_batch.config import resolve_config
config = resolve_config()  # File values fill in when env is absent
print(f"Extra fields: {config.extra}")  # Validated extra fields
```

Precedence reminder: Programmatic > Env > Project file > Home file > Defaults. For full rules and file discovery options, see Reference → [Configuration](../reference/configuration.md).

---

## 3) Profiles (per‑environment presets)

Goal: Switch between presets without changing code.

```toml
# pyproject.toml
[tool.gemini_batch.profiles.dev]
model = "gemini-2.0-flash"
use_real_api = false
experimental_features = true  # experimental_* pattern

[tool.gemini_batch.profiles.prod]
model = "gemini-2.0-pro"
use_real_api = true
```

Select a profile at runtime:

```bash
export GEMINI_BATCH_PROFILE=prod
```

Or programmatically:

```python
config = resolve_config(profile="dev")
```

Tip: Use a personal file at `~/.config/gemini_batch.toml` for machine‑specific defaults; the project file still wins over the home file.

---

## 4) Programmatic overrides (highest priority)

Goal: Pin values for a specific run or test.

```python
from gemini_batch.config import resolve_config

# Direct overrides
config = resolve_config(overrides={
    "model": "gpt-4o",  # Will infer provider="openai"
    "use_real_api": False,
})

# With audit trail
config, sources = resolve_config(
    overrides={"model": "claude-3-5-sonnet"},
    explain=True
)
print(f"Model came from: {sources['model'].origin}")
```

Any key provided here overrides env and files for that run.

---

## 5) Scoped overrides in tests

Goal: Temporarily adjust configuration inside a block (async‑safe).

```python
from gemini_batch.config import config_scope, resolve_config

# Context manager for isolated testing
with config_scope({"use_real_api": False, "model": "test-model"}):
    # Inside: resolution will see test values
    test_config = resolve_config()
    run_test_suite()
# Outside: original configuration restored
```

> The scope affects resolution time only. Once resolved, a `FrozenConfig` snapshot is used; handlers won't observe ambient changes.

---

## 6) Audit and debug configuration

Goal: Understand why values are what they are.

```python
from gemini_batch.config import (
    resolve_config,
    check_environment,
    doctor,
    audit_text,
    audit_layers_summary,
)

# Check environment variables (both GEMINI_* and GEMINI_BATCH_ are listed; secrets redacted)
env_vars = check_environment()
print("Environment snapshot:", env_vars)

# Get diagnostic information
for message in doctor():
    print(f"🔍 {message}")

# Redacted audit (human-readable origin labels)
cfg, src = resolve_config(explain=True)
print(audit_text(cfg, src))
for line in audit_layers_summary(src):
    print(line)

# If you need raw origin enums per field:
for field, fo in src.items():
    print(field, fo.origin.value)
```

See also: Reference → CLI for command equivalents and quick checks:

- `gb-config show` (effective redacted config)
- `gb-config audit` (field origins + layer summary)
- `gb-config doctor` (actionable messages)
- `gb-config env` (redacted environment snapshot)

For full diagnostics, provider inference rules, precedence, file discovery, and extra‑field validation patterns, see Reference → [Configuration](../reference/configuration.md).

Conventional patterns:

- `*_timeout` (int) — Timeout values in seconds
- `*_url` (str) — URL endpoints for external services
- `experimental_*` (Any) — Experimental features (unstable API)
- `legacy_*` (Any, deprecated) — Triggers deprecation warnings

---

## 7) Home‑level defaults

Goal: Provide personal defaults across projects.

Create `~/.config/gemini_batch.toml`:

```toml
[tool.gemini_batch]
model = "gemini-2.0-flash"
tier = "free"

[tool.gemini_batch.profiles.personal]
model = "gemini-2.0-pro"
experimental_features = true
```

These values are lower priority than the project file and env, but higher than built‑in defaults.

Path override (advanced):

- Set `GEMINI_BATCH_CONFIG_HOME` to an alternate file path (e.g., `/tmp/gemini_batch.toml`). Helpful in containers or tests.

---

## 8) Debug audit emission

Goal: Enable redacted debug audit output for troubleshooting.

```bash
export GEMINI_BATCH_DEBUG_CONFIG=1
```

```python
from gemini_batch.config import resolve_config

# First call emits debug audit (prints once per callsite by default)
config1 = resolve_config()

# Subsequent calls don't emit (already done)
config2 = resolve_config()

# CLI equivalent for quick checks:
#   gb-config doctor
#   gb-config audit
```

Debug emission is controlled by `GEMINI_BATCH_DEBUG_CONFIG` and uses Python’s warnings; audits are redacted.
