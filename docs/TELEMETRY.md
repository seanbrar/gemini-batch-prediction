# Telemetry Integration

The `gemini-batch` library includes a `TelemetryContext` for advanced metrics collection. You can integrate it with your own metrics systems (e.g., Prometheus, DataDog) by creating a custom reporter.

This feature is designed for production environments where detailed telemetry is required. For internal debugging tools, see `dev/telemetry_internals.md`.

## Implementing a Custom Reporter

To create a custom reporter, implement a class with `record_timing` and/or `record_metric` methods. The library uses duck-typing, so no base class is required.

```python
# In your application's telemetry module
import logging

class MyCustomReporter:
    """A simple reporter that logs to a dedicated logger."""
    def __init__(self):
        self.logger = logging.getLogger("gemini_telemetry")
        self.logger.setLevel(logging.INFO)
        
        # Configure a handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - METRIC: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def record_timing(self, scope: str, duration: float, **metadata):
        """Records a timing metric."""
        self.logger.info(
            "TIMING: scope=%s, duration=%.4fs", 
            scope, 
            duration
        )
    
    def record_metric(self, scope: str, value, **metadata):
        """Records a value-based metric."""
        metric_type = metadata.get('metric_type', 'gauge')
        self.logger.info(
            "METRIC: scope=%s, value=%s, type=%s", 
            scope, 
            value, 
            metric_type
        )

```

## Integrating the Reporter

Inject your custom reporter into the `BatchProcessor` via the `TelemetryContext`.

```python
# In your main application
from gemini_batch import BatchProcessor, TelemetryContext
# from my_app.telemetry import MyCustomReporter

# 1. Instantiate your reporter
my_reporter = MyCustomReporter()

# 2. Create a TelemetryContext with one or more reporters
tele_context = TelemetryContext(my_reporter)

# 3. Inject it into the BatchProcessor
processor = BatchProcessor(telemetry_context=tele_context)

# Now, all operations within the processor will send metrics
# to your custom reporter.
content = "Machine learning enables pattern recognition in data."
questions = ["What is machine learning?", "What does it enable?"]

results = processor.process_questions(content, questions)
```

## Advanced Integration

The `TelemetryReporter` protocol is designed to be flexible:

- **Multiple Reporters**: You can pass multiple reporters to `TelemetryContext`: `TelemetryContext(reporter1, reporter2)`.
- **"Good Citizen" Design**: The system does not impose any external library dependencies, allowing you to use your existing monitoring clients.
- **Detailed Metadata**: The `metadata` dictionary passed to your reporter includes contextual information like `depth` and `parent_scope` for more sophisticated analysis.

For a complete, runnable example, see `examples/custom_telemetry_reporter.py`.
