# Telemetry Integration

The library includes a `TelemetryContext` for advanced metrics collection. You can integrate it with your own monitoring systems (e.g., Prometheus, DataDog) by creating a custom reporter.

This feature is designed for production environments where detailed telemetry is required. For internal debugging tools and design philosophy, see the [Developer Guide](../dev/telemetry_internals.md).

-----

## Implementing a Custom Reporter

You can create a custom reporter by creating a class that implements `record_timing` and/or `record_metric` methods.

While the system uses structural typing (meaning no base class is technically required), it's **highly recommended** that your reporter inherits from the `TelemetryReporter` protocol. This enables static analysis tools and IDEs to provide autocompletion and type-checking, ensuring your implementation is correct.

```python
from your_module.telemetry import TelemetryReporter

# A simple reporter that logs to the console.
# Inheriting from TelemetryReporter provides IDE support.
class MyCustomReporter(TelemetryReporter):
    def record_timing(self, scope: str, duration: float, **metadata):
        print(f"[METRIC] TIMING: scope={scope}, duration={duration:.4f}s")

    def record_metric(self, scope: str, value: any, **metadata):
        # The 'metric_type' is included in the metadata for counters and gauges.
        metric_type = metadata.get('metric_type', 'unknown')
        print(f"[METRIC] METRIC: scope={scope}, value={value}, type={metric_type}")
```

-----

## Integrating the Reporter

Inject your custom reporter into services like `BatchProcessor` by wrapping it in a `TelemetryContext`.

```python
from gemini_batch import BatchProcessor, TelemetryContext
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
* **Detailed Metadata**: The `metadata` dictionary passed to your reporter includes valuable context like `depth` and `parent_scope` for more sophisticated analysis.
