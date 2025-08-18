# Telemetry Integration

> Note: This page describes the current API. For the upcoming architecture, see Explanation → Command Pipeline.

The library includes a `TelemetryContext` for advanced metrics collection. You can integrate it with your own monitoring systems (e.g., Prometheus, DataDog) by creating a custom reporter.

This feature is designed for production environments where detailed telemetry is required. For design rationale and implementation details, see [Explanation → Concepts (Telemetry)](./explanation/concepts/telemetry.md), [Deep Dives → Telemetry Spec](./explanation/deep-dives/telemetry-spec.md), and [Decisions → ADR-0006 Telemetry](./explanation/decisions/ADR-0006-telemetry.md).

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

Inject your custom reporter into services like `BatchProcessor` by wrapping it in a `TelemetryContext`.

```python
from gemini_batch import BatchProcessor
from gemini_batch.telemetry import TelemetryContext
# from my_app.telemetry import MyCustomReporter

# 1. Instantiate your reporter and the TelemetryContext factory
my_reporter = MyCustomReporter()
tele_context = TelemetryContext(my_reporter)

# 2. Inject the context into the BatchProcessor
processor = BatchProcessor(telemetry_context=tele_context)

# All operations within the processor will now send metrics to your reporter.
results = processor.process_questions(...)
```

-----

## Advanced Integration

The `TelemetryReporter` protocol is designed for flexibility:

* **Multiple Reporters**: You can pass several reporters to the `TelemetryContext` factory, and each will receive all telemetry events.
* **"Good Citizen" Design**: The system doesn't impose external library dependencies, allowing you to use your existing monitoring clients.
* **Event Metadata**:
  * Timing events include: `depth`, `parent_scope`, `call_count`, and timing provenance (`start_monotonic_s`, `end_monotonic_s`, `start_wall_time_s`, `end_wall_time_s`).
  * Metric events include: `depth`, `parent_scope`. When using `count(...)` or `gauge(...)`, `metric_type` is set accordingly.

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
export GEMINI_TELEMETRY_STRICT_SCOPES=1
```

Valid examples: `pipeline.plan.tokens`, `api.request.send`, `batch.process.step_1`

Invalid examples: `Pipeline.Plan` (uppercase), `api/request` (slash), `step-1` (dash)

### Feature Flags and Developer Experience

* Telemetry is enabled only when BOTH are true:
  * An environment flag is set: `GEMINI_TELEMETRY=1` or `DEBUG=1`.
  * You pass at least one reporter to `TelemetryContext(...)`.
* If no reporters are provided, the factory returns a no-op singleton, even if the env flag is set.
* Check `tele.is_enabled` in hot paths to skip expensive metadata collection.

### Deprecated helper

`tele_scope(ctx, name, **metadata)` remains as a deprecated alias for calling the context directly. Prefer `with tele(name):`.
