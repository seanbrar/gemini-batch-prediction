# Developer Guide: Telemetry Internals

**Audience:** Internal Developers, Contributors
**Purpose:** To document the design, philosophy, and internal tooling of the telemetry system. This guide ensures that contributions are instrumented consistently.

## 1. Core Philosophy: Minimal Overhead and "Clever Simplicity"

The telemetry system is designed around a critical principle: **instrumentation must have negligible overhead when disabled**. This is achieved by using a factory pattern that, at startup, decides whether to provide a full-featured telemetry context or a completely inert, singleton no-op object.

The core tenets of this system are:

* **Contextual Intelligence**: Automatically track nested scopes and enrich metrics with context using thread-safe `ContextVar` variables.
* **Extensibility via Protocols**: Allow users to integrate their own monitoring tools without rigid inheritance.
* **Negligible Overhead Guarantee**: Ensure that when telemetry is disabled, the performance impact is virtually zero by returning a no-op object.

-----

## 2. Architectural Design

The architecture is composed of the `TelemetryContext` factory (the entry point), two underlying context objects (`_EnabledTelemetryContext` and `_NoOpTelemetryContext`), and the `TelemetryReporter` protocol (the extensible reporting mechanism).

### 2.1. The `TelemetryContext`: Central Factory

The main entry point for all instrumentation is the `TelemetryContext` factory. A developer gets a context object from this factory, which manages the scope stack, timing, and dispatching of metrics to all registered reporters.

Its primary interface is making the context object *callable*:

```python
# The primary interaction pattern
with self.tele("batch.total_processing", question_count=len(questions)):
    # ... code to be timed ...
    self.tele.gauge("token_efficiency", some_value)
```

This automatically handles timing, duration calculation, and hierarchical scope naming (e.g., `batch.total_processing.attempt`), and enriches data with metadata like call depth and parent scope.

### 2.2. The `TelemetryReporter` Protocol: "Good Citizen" Extensibility

We use a `typing.Protocol` instead of a traditional abstract base class. This allows any object that "looks like" a reporter (i.e., has the right methods) to be used, decoupling the library from user-specific implementations.

```python
@runtime_checkable
class TelemetryReporter(Protocol):
    """Duck-typed protocol - no inheritance required"""
    def record_timing(self, scope: str, duration: float, **metadata) -> None: ...
    def record_metric(self, scope: str, value: Any, **metadata) -> None: ...

class MyCustomReporter:
    # No inheritance from TelemetryReporter needed!
    def record_timing(self, scope: str, duration: float, **metadata):
        my_metrics_service.timer(scope, duration)

# Works seamlessly because it matches the protocol's "shape"
tele_context = TelemetryContext(MyCustomReporter())
```

-----

## 3. The Ultra-Low Overhead Design

The system's performance guarantee comes from the factory pattern, which is evaluated once based on environment variables.

### The No-Op Singleton Pattern

The `TelemetryContext` factory provides the core optimization. It checks the `_TELEMETRY_ENABLED` flag (set by `GEMINI_TELEMETRY=1` or `DEBUG=1`) *once*.

* If **enabled**, it returns a full-featured `_EnabledTelemetryContext`.
* If **disabled**, it returns `_NO_OP_SINGLETON`, a single, shared, immutable instance of `_NoOpTelemetryContext`.

```python
# A single, stateless, immutable no-op object is created once
_NO_OP_SINGLETON = _NoOpTelemetryContext()

def TelemetryContext(*reporters: TelemetryReporter) -> TelemetryContextProtocol:
    """
    Factory that returns either a full-featured telemetry context or a
    single, shared, no-op instance for maximum performance when disabled.
    """
    # This check happens when the context is created, not on every call
    if _TELEMETRY_ENABLED and reporters:
        return _EnabledTelemetryContext(*reporters)
    else:
        # When disabled, always return the same, pre-existing no-op instance.
        return _NO_OP_SINGLETON
```

The methods on `_NoOpTelemetryContext` do nothing, so calls like `tele("my_op")` or `tele.metric(...)` become completely inert with no conditional checks, providing a negligible performance cost.

-----

## 4. Internal Tooling: The `_SimpleReporter`

The library includes `_SimpleReporter`, an in-memory reporter for internal development and debugging. **It is not intended as a primary user-facing feature.** Its purpose is to provide a quick way to inspect telemetry during development.

### How to Use `_SimpleReporter` for Debugging

The reporter is only active when `GEMINI_TELEMETRY=1` or `DEBUG=1` is set.

1. **Set the environment variable**:

    ```bash
    export GEMINI_TELEMETRY=1
    ```

2. **Instantiate and inject it into the `TelemetryContext` factory**:
    In your test or script, create a `_SimpleReporter` and pass it to the factory.

    ```python
    # Note the direct import of the internal class
    from your_module.telemetry import _SimpleReporter, TelemetryContext

    # 1. Create the reporter instance
    reporter = _SimpleReporter()

    # 2. Pass it to the factory to get an enabled context
    tele_context = TelemetryContext(reporter)

    # 3. Use the context in your code
    with tele_context("test.operation"):
        tele_context.count("items_processed")

    # 4. Print the collected data
    reporter.print_report()
    ```
