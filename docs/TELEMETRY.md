# Telemetry Integration

The `gemini-batch` library includes a `TelemetryContext` for advanced metrics collection. You can integrate it with your own metrics systems (e.g., Prometheus, DataDog) by creating a custom reporter.

This feature is designed for production environments where detailed telemetry is required. For internal debugging tools, see `dev/telemetry_internals.md`.

## Implementing a Custom Reporter

To create a custom reporter, implement a class with `record_timing` and/or `record_metric` methods. The library uses duck-typing, so no base class is required.

```python
# A simple reporter that logs to the console
class MyCustomReporter:
    def record_timing(self, scope: str, duration: float, **metadata):
        print(f"[METRIC] TIMING: scope={scope}, duration={duration:.4f}s")

    def record_metric(self, scope: str, value, **metadata):
        metric_type = metadata.get('metric_type', 'gauge')
        print(f"[METRIC] GAUGE: scope={scope}, value={value}, type={metric_type}")
```

## Integrating the Reporter

Inject your custom reporter into the `BatchProcessor` via the `TelemetryContext`.

```python
from gemini_batch import BatchProcessor, TelemetryContext
# from my_app.telemetry import MyCustomReporter

# 1. Instantiate your reporter and context
my_reporter = MyCustomReporter()
tele_context = TelemetryContext(my_reporter)

# 2. Inject it into the BatchProcessor
processor = BatchProcessor(telemetry_context=tele_context)

# Now, all operations will send metrics to your custom reporter.
results = processor.process_questions(...)
```

## Production Integration Example: Prometheus

To demonstrate a real-world use case, the framework includes a complete example for sending telemetry to a Prometheus server. This shows how to map the library's timing and metric events to Prometheus counters and histograms.

See [`examples/prometheus_telemetry_demo.py`](../examples/prometheus_telemetry_demo.py) for the full implementation.

## Advanced Integration

The `TelemetryReporter` protocol is designed to be flexible:

- **Multiple Reporters**: You can pass multiple reporters to `TelemetryContext`: `TelemetryContext(reporter1, reporter2)`.
- **"Good Citizen" Design**: The system does not impose any external library dependencies, allowing you to use your existing monitoring clients.
- **Detailed Metadata**: The `metadata` dictionary passed to your reporter includes contextual information like `depth` and `parent_scope` for more sophisticated analysis.

For a complete, runnable example, see `examples/custom_telemetry_reporter.py`.
