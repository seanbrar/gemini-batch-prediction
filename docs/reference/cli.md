# CLI — gb-config

Minimal CLI for inspecting and validating effective configuration. Useful for local checks, CI logs, and quick diagnostics.

## Overview

- Command: `gb-config`
- Availability: installed with the package (console script)
- Redaction: secrets are never printed; sensitive values appear as `***redacted***`.

## Commands

- `gb-config show`: Prints the effective, redacted config as JSON.
  - Use when you need a machine-readable snapshot for logs or CI artifacts.
  - Example:

    ```bash
    gb-config show
    # {
    #   "model": "gemini-2.0-flash",
    #   "api_key": "***redacted***",
    #   "use_real_api": false,
    #   "enable_caching": false,
    #   "ttl_seconds": 3600,
    #   "telemetry_enabled": false,
    #   "tier": "free",
    #   "provider": "google",
    #   "extra": {}
    # }
    ```

- `gb-config audit`: Prints a human-readable audit and a layer summary.
  - Shows each field’s origin (default, home, project, env, overrides).
  - Example (truncated):

    ```bash
    gb-config audit
    # model: gemini-2.0-flash (default)
    # tier: free (default)
    # ...
    # default  : N fields
    # home     : 0 fields
    # project  : 0 fields
    # env      : M fields
    # overrides: 0 fields
    ```

- `gb-config doctor`: Prints actionable messages and advisories.
  - Use to confirm readiness before running real API calls.
  - Example:

    ```bash
    gb-config doctor
    # No issues detected.
    # Advisory: 'tier' not specified; using default FREE. See: tier (enum: FREE|TIER_1|...)
    ```

- `gb-config env`: Prints relevant environment variables (redacted where needed).
  - Includes both `GEMINI_BATCH_*` and `GEMINI_*` variables.
  - Example:

    ```bash
    gb-config env | sort
    # GEMINI_API_KEY=***redacted***
    # GEMINI_BATCH_TIER=free
    # GEMINI_BATCH_USE_REAL_API=1
    ```

## Tips

- CI usage: `gb-config show` and `gb-config doctor` always exit `0`; parse output for warnings or raise in your pipeline if necessary.
- Redaction: `api_key` and other sensitive fields are always redacted by design.
- See also: How‑to → Configuration for precedence rules and audit details.

## CI usage examples

Shell (fail build if API key required but missing):

```bash
if gb-config doctor | grep -q "api_key is missing"; then
  echo "❌ Missing API key while use_real_api=True" >&2
  exit 1
fi
```

Python (doctest‑style):

```pycon
>>> import subprocess
>>> out = subprocess.check_output(["gb-config", "doctor"], text=True)
>>> "api_key is missing" not in out
True
```

Last reviewed: 2025-09
