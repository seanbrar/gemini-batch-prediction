# Troubleshooting

Common issues and quick fixes when getting started.

## Real API enabled but no key

- Symptom: `gb-config doctor` prints `use_real_api=True but api_key is missing.` or calls behave like mock.
- Fix: Set `GEMINI_API_KEY` and enable real API, or disable real API.
  - `export GEMINI_API_KEY="<your key>"`
  - `export GEMINI_BATCH_USE_REAL_API=1` (or unset to stay in mock mode)

## Hitting rate limits immediately

- Symptom: Slow or throttled requests after enabling real API.
- Fix: Set your billing tier to match your account; optionally reduce fan‑out.
  - `export GEMINI_BATCH_TIER=free|tier_1|tier_2|tier_3`
  - Lower concurrency for vectorized calls via config or options
    - Config: `request_concurrency` in your config file or env
    - Per call: `make_execution_options(request_concurrency=1)`

## Model/provider mismatch

- Symptom: `Unknown model; provider defaulted to 'google'.`
- Fix: Use a valid model string for your provider (e.g., `gemini-2.0-flash`).

## “Mock” answers when expecting real results

- Symptom: Outputs look like `echo: ...` or include `mock` metadata.
- Fix: Ensure real API is enabled and key is present:
  - `export GEMINI_BATCH_USE_REAL_API=1`
  - `export GEMINI_API_KEY="<your key>"`
  - Re‑run `gb-config doctor` to confirm.

## Secrets appear missing in logs

- Symptom: Keys are printed as `***redacted***`.
- Explanation: Redaction is by design for safety. Use `gb-config env` to confirm variables exist (still redacted), or rely on `gb-config doctor` and application behavior.

## Python or import errors

- Symptom: `ModuleNotFoundError` or runtime errors on import.
- Fixes:
  - Use Python `3.13` and a clean virtual environment.
  - Install from Releases (wheel) or source with `pip install -e .`.

## Notebook visualization missing

- Symptom: Import errors for plotting in notebooks.
- Fix: Install visualization helpers:
  - `pip install "matplotlib~=3.10" "pandas~=2.3" "seaborn~=0.13"`

## Still stuck?

- Run: `gb-config show` and `gb-config doctor` and attach output to any issue report.
- See also: How‑to → FAQ; How‑to → Configuration; How‑to → Logging; Reference → CLI.
