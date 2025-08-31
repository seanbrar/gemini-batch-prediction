Metrics in ResultEnvelope
=========================

Overview
--------

- Purpose: Concise runtime stats that are safe to surface as part of results. Distinct from telemetry (which is richer and often ephemeral/dev‑only).
- Source: Returned on every pipeline run via the `ResultEnvelope` under the `metrics` and `usage` keys.
- Units: Durations are recorded in seconds (float) using `time.perf_counter()`; convert to ms for display if desired.

What’s Included
---------------

- metrics.durations: Map of pipeline stage → seconds. Example keys include `SourceHandler`, `ExecutionPlanner`, `CacheStage`, `APIHandler`, `RateLimitHandler`, `ResultBuilder`.
- metrics.token_validation: When token estimation is available, contains `estimated_min`, `estimated_expected`, `estimated_max`, `actual`, and `in_range`.
- metrics.hints: Optional flags (e.g., `prefer_json_array`) used during extraction.
- metrics.per_prompt (vectorized batches): Per‑prompt usage/meta records (attached by the API handler as telemetry and surfaced by the builder when present).
- metrics.per_call_meta (vectorized batches): Per‑call metadata including `duration_s` (time spent inside the concurrency semaphore), `api_time_s` (provider API time for successful attempt[s]), `non_api_time_s` (all non-API time such as client overhead, queuing, and backoff wait time), and `cache_applied` flag.
- metrics.concurrency_used: Integer semaphore size used for the vectorized calls.
- metrics.cache_application: How cache names were applied for this plan: `"none"`, `"plan"`, or `"override"`.
- metrics.per_call_estimates: When available, a tuple of estimates with `min_tokens`, `expected_tokens`, `max_tokens`, and `confidence` for each call.
- usage: Token usage summary when provided (e.g., `input_tokens`, `output_tokens`, `total_tokens` or `total_token_count`).

How to Retrieve
---------------

- From frontdoor helpers (`run_simple`, `run_batch`):
  - Await the call to get a `ResultEnvelope` dictionary. Read `env["metrics"]` and `env["usage"]`.
- From executor (`GeminiExecutor.execute`):
  - The returned value is a `ResultEnvelope`. Same access pattern as above.
- From conversation extension (`Conversation.run`):
  - Returns `(Conversation, answers, BatchMetrics)`. `BatchMetrics` includes numeric totals and per‑prompt aggregates derived from `usage`/`metrics` but does not include stage durations.

Examples
--------

- Get stage timings (seconds → ms):

  ```python
  env = await run_batch(["Q1", "Q2"], sources=[src])
  durs = (env.get("metrics") or {}).get("durations") or {}
  for stage, secs in sorted(durs.items(), key=lambda kv: kv[1], reverse=True):
      print(f"{stage}: {secs*1000:.1f} ms")
  ```

- Token validation:

  ```python
  tv = (env.get("metrics") or {}).get("token_validation") or {}
  if tv:
      print("Estimated:", tv.get("estimated_expected"), "Actual:", tv.get("actual"), "In range:", tv.get("in_range"))
  ```

- Usage totals (shape may vary by provider/mock):

  ```python
  usage = env.get("usage") or {}
  total = usage.get("total_tokens") or usage.get("total_token_count")
  print("Total tokens:", total)
  ```

- Per-call metadata and concurrency:

  ```python
  m = env.get("metrics") or {}
  print("concurrency:", m.get("concurrency_used"))
  for i, meta in enumerate(m.get("per_call_meta") or ()):  # tuple of dicts
      print(i,
            "duration_s:", meta.get("duration_s"),
            "api_time_s:", meta.get("api_time_s"),
            "non_api_time_s:", meta.get("non_api_time_s"),
            "cache_applied:", meta.get("cache_applied"))
  ```

Precision & Formatting
----------------------

- Durations are small in mock/dev flows. If displayed with low precision (e.g., `:.1f` seconds), they can round to `0.0`.
- Prefer ms for readability: `secs * 1000` with one decimal place, or use `:.6f` seconds for sub‑ms precision.

Common Pitfalls
---------------

- Unit mismatch: Treating seconds as milliseconds (or vice‑versa). Always convert before printing.
- Rounding to zero: Formatting tiny second values with too few decimals yields `0.0`. Increase precision or display in ms.
- Assuming presence: `usage` and some `metrics` fields are optional depending on provider and plan; guard with defaults.

Notes
-----

- The executor guarantees a `ResultEnvelope` and attaches per‑stage durations even when a custom terminal stage is used.
- For deep, dev‑time telemetry (scoped timings, counters), use the `TelemetryContext`. Those are not part of the stable `ResultEnvelope` API.
