# Telemetry Integration

Last reviewed: 2025-09

> Note: This page describes the current API. For the upcoming architecture, see Explanation → Command Pipeline.

The library includes a `TelemetryContext` for advanced metrics collection. You can integrate it with your own monitoring systems (e.g., Prometheus, DataDog) by creating a custom reporter.

## Prerequisites

- Python 3.13 and `gemini_batch` installed.
- Basic logging or a monitoring client available (for examples below, Python `logging`).
- Optional: set `GEMINI_BATCH_TELEMETRY=1` to enable emission; otherwise the context becomes a no‑op.

This feature is designed for production environments where detailed telemetry is required. For design rationale and implementation details, see [Explanation → Concepts (Telemetry)](./explanation/concepts/telemetry.md), [Deep Dives → Telemetry Spec](./explanation/deep-dives/telemetry-spec.md), and [Decisions → ADR-0006 Telemetry](./explanation/decisions/ADR-0006-telemetry.md).

## Raw Preview (Research)

For researcher workflows, you can attach a compact, sanitized preview of the raw provider output to telemetry. This preview is opt-in and designed to be tiny and safe for triage.

- Flag: set `GEMINI_BATCH_TELEMETRY_RAW_PREVIEW=1` to enable globally, or pass `include_raw_preview=True` to `APIHandler(...)` for local control.
- Location: the preview is attached to the finalized command telemetry and surfaced in the `ResultEnvelope` under `metrics.raw_preview`.
- Fields: a minimal set such as `model`, `text`, `candidate0_text`, `finish_reason`, and a sanitized `usage` (scalar-only; long strings truncated). Unknown shapes fall back to a truncated `repr`.

Example (enabling via constructor):

```python
from gemini_batch.pipeline.api_handler import APIHandler

handler = APIHandler(include_raw_preview=True)  # opt-in per handler
# ... run the pipeline (planner → api handler → result builder)
envelope = result["value"]  # ResultEnvelope from ResultBuilder
raw_preview = envelope["metrics"]["raw_preview"]
print(raw_preview)
# {
#   "model": "gemini-2.0-flash",
#   "batch": (
#       {"text": "...", "candidate0_text": "...", "finish_reason": "STOP", "usage": {"total_token_count": 123}},
#       {"text": "...", "candidate0_text": "...", "finish_reason": "STOP", "usage": {"total_token_count": 140}},
#   )
# }
```

Example (enabling via environment):

```bash
export GEMINI_BATCH_TELEMETRY_RAW_PREVIEW=1
python your_pipeline_script.py
```

Notes:

- The preview contains only small fields (e.g., `model`, `text`, `candidate0_text`, `finish_reason`, and a sanitized `usage`) and truncates long strings.
- `usage` is sanitized to include only simple scalar fields; nested structures are dropped.
- It is attached best-effort and never fails the pipeline if extraction fails.
- Vectorized calls expose a tuple of per-call previews under `batch`.

-----

## Quick Start (Production-Oriented)

Most users should either monitor logs or emit telemetry to an external backend. The built-in `_SimpleReporter` is useful for development, but is not recommended for regular use.

```python
import os
import logging
from typing import Any
from gemini_batch.telemetry import TelemetryContext, TelemetryReporter

# 1) Enable telemetry via env (opt-in)
os.environ["GEMINI_BATCH_TELEMETRY"] = "1"

# 2) Example: bridge timings/metrics to your existing logging setup
class LoggingReporter:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("gemini_batch.telemetry")

    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        # Keep it compact and structured for downstream log processors
        self.log.info("telemetry.timing", extra={
            "scope": scope, "duration_s": duration, **metadata,
        })

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        self.log.info("telemetry.metric", extra={
            "scope": scope, "value": value, **metadata,
        })

# 3) Install reporter
tele = TelemetryContext(LoggingReporter())

# 4) Your application code (the library emits scopes internally)
with tele("my.pipeline.step", batch_size=16):
    tele.gauge("token_efficiency", 0.92)
```

Success check:

- Expect INFO log lines containing `telemetry.timing` and `telemetry.metric` with your scopes and metadata.
- Programmatic check: `assert tele.is_enabled` and wrap a dummy scope to ensure your reporter receives events.

Notes:

- Prefer sending telemetry to a production backend (e.g., Prometheus, OpenTelemetry collector) rather than relying on a custom in-process reporter.
- If you only need human inspection, INFO logs with structured fields are often sufficient.

-----

## Implementing a Custom Reporter

You can create a custom reporter by creating a class that implements `record_timing` and/or `record_metric` methods.

Recommended — Structural typing (no inheritance):

```python
from typing import Any
from gemini_batch.telemetry import TelemetryContext, TelemetryReporter

class MyCustomReporter:
    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        print(f"[TIMING] {scope} took {duration:.4f}s")

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        print(f"[METRIC] {scope} = {value} ({metadata})")

reporter: TelemetryReporter = MyCustomReporter()  # Optional type annotation for IDEs/mypy
tele = TelemetryContext(reporter)
```

Optional — Runtime conformance check (when you accept reporters from third parties):

```python
from typing import Any
from gemini_batch.telemetry import TelemetryContext, TelemetryReporter

class MyCustomReporter:
    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None: ...
    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None: ...

reporter = MyCustomReporter()
if not isinstance(reporter, TelemetryReporter):  # TelemetryReporter is @runtime_checkable
    raise TypeError("Reporter does not conform to TelemetryReporter protocol")

tele = TelemetryContext(reporter)
```

-----

## Integrating the Reporter

There are two ways to capture pipeline telemetry.

Option A — Enable built‑in telemetry via environment flags (no custom reporter):

- Set `GEMINI_BATCH_TELEMETRY=1`. The library attaches a tiny, internal reporter and surfaces metrics into the `ResultEnvelope` (under `metrics` and `usage`).
- Read `env["metrics"]`/`env["usage"]` from the result. See Reference → ResultEnvelope Metrics for shapes.

Option B — Provide your own reporter to the API handler in a custom pipeline:

```python
from __future__ import annotations
import logging
from typing import Any

from gemini_batch.executor import GeminiExecutor
from gemini_batch.config import resolve_config
from gemini_batch.pipeline.source_handler import SourceHandler
from gemini_batch.pipeline.planner import ExecutionPlanner
from gemini_batch.pipeline.remote_materialization import RemoteMaterializationStage
from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler
from gemini_batch.pipeline.cache_stage import CacheStage
from gemini_batch.pipeline.api_handler import APIHandler
from gemini_batch.pipeline.result_builder import ResultBuilder
from gemini_batch.telemetry import TelemetryContext

class LoggingReporter:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("gemini_batch.telemetry")
    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        self.log.info("telemetry.timing", extra={"scope": scope, "duration_s": duration, **metadata})
    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        self.log.info("telemetry.metric", extra={"scope": scope, "value": value, **metadata})

cfg = resolve_config()
tele = TelemetryContext(LoggingReporter())  # requires GEMINI_BATCH_TELEMETRY=1

handlers = [
    SourceHandler(),
    ExecutionPlanner(),
    RemoteMaterializationStage(),
    RateLimitHandler(),
    CacheStage(registries={}, adapter_factory=None),
    APIHandler(telemetry=tele, registries={"cache": None, "files": None}, adapter_factory=None),
    ResultBuilder(),
]

executor = GeminiExecutor(cfg, pipeline_handlers=handlers)
```

Notes

- Option A is simplest and surfaces metrics in the result envelope for immediate use.
- Option B streams timings/metrics to your backend via a reporter; it requires a custom executor and `GEMINI_BATCH_TELEMETRY=1` to enable emission.

Verification

- Option A: run any `run_simple`/`run_batch` call and inspect `env["metrics"]` and `env["usage"]`.
- Option B: expect INFO log lines (or your backend events) with `telemetry.timing` and `telemetry.metric`.

-----

## Privacy and Data Handling

!!! warning "Handle data responsibly"
    - Avoid logging raw inputs, prompts, or secrets. Prefer IDs, short labels, and small scalar metadata in telemetry and logs.
    - Raw preview is sanitized and truncated by design, but still opt‑in. Enable only when necessary for triage and disable in steady‑state production.
    - Redaction: API keys and sensitive fields are never printed by `gb-config`; apply the same discipline to your reporters and log processors.
    - Minimize retention: if exporting telemetry, keep payloads small and set retention appropriate to your compliance requirements.
    - PII: If processing personal data, align with your org’s policies (e.g., GDPR/CCPA). Do not include user content in telemetry; prefer anonymized counters and bounded metrics.

## Advanced Integration

The `TelemetryReporter` protocol is designed for flexibility:

- **Multiple Reporters**: You can pass several reporters to the `TelemetryContext` factory, and each will receive all telemetry events.
- **"Good Citizen" Design**: The system doesn't impose external library dependencies, allowing you to use your existing monitoring clients.
- **Event Metadata**:
  - Timing events include: `depth`, `parent_scope`, `call_count`, and timing provenance (`start_monotonic_s`, `end_monotonic_s`, `start_wall_time_s`, `end_wall_time_s`).
  - Metric events include: `depth`, `parent_scope`. When using `count(...)` or `gauge(...)`, `metric_type` is set accordingly.

### Optional Raw Preview (Debug)

For manual triage, you can opt in to attach a compact, sanitized preview of the raw provider response into the result envelope under `metrics.raw_preview`.

Enable globally via env:

```bash
export GEMINI_BATCH_TELEMETRY_RAW_PREVIEW=1
```

Or enable per handler in code:

```python
from gemini_batch.pipeline.api_handler import APIHandler

handler = APIHandler(include_raw_preview=True)
```

Notes:

- Disabled by default to keep envelopes lean and avoid leaking large payloads.
- The preview truncates long strings and includes only a few common fields.
- `usage` is sanitized (scalar-only; truncated strings; nested structures omitted).
- Prefer production telemetry backends for ongoing analysis; this is a convenience for researchers.

Safety:

- Redaction/truncation prevents large payloads or sensitive content from being emitted.
- Do not attach raw inputs or secrets to telemetry metadata; prefer IDs and stable labels.

### Convenience Methods

The context offers convenience helpers in addition to the primary `with tele("scope"):` usage:

```python
# Timing (alias of calling the context)
with tele.time("pipeline.plan"):
    ...

# Counters and gauges (metadata is optional)
tele.count("api.requests", increment=1, component="planner")
tele.gauge("batch.token_efficiency", value=0.87)
```

### Scope Naming and Strict Validation

Scopes are dot-separated lowercase tokens (e.g., `api.request.send`). To optionally enforce a strict naming regex at runtime, set:

```bash
export GEMINI_BATCH_TELEMETRY_STRICT_SCOPES=1
```

Valid examples: `pipeline.plan.tokens`, `api.request.send`, `batch.process.step_1`

Invalid examples: `Pipeline.Plan` (uppercase), `api/request` (slash), `step-1` (dash)

### Feature Flags and Developer Experience

- Telemetry is enabled only when BOTH are true:
  - An environment flag is set: `GEMINI_BATCH_TELEMETRY=1` or `DEBUG=1`.
  - You pass at least one reporter to `TelemetryContext(...)`.
- If no reporters are provided, the factory returns a no-op singleton, even if the env flag is set.
- Check `tele.is_enabled` in hot paths to skip expensive metadata collection.

### Deprecated helper

`tele_scope(ctx, name, **metadata)` remains as a deprecated alias for calling the context directly. Prefer `with tele(name):`.
