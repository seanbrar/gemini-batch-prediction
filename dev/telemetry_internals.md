# Developer Guide: Telemetry Internals

**Audience:** Internal Developers, Contributors
**Purpose:** To document the design, philosophy, and internal tooling of the telemetry system. This guide ensures that contributions are instrumented consistently.

## 1. Core Philosophy: Zero-Overhead and "Clever Simplicity"

The telemetry system is designed around a critical principle: **instrumentation must have zero-overhead when disabled**. This is achieved through a compile-time flag that removes the instrumentation code entirely when not in use, ensuring production code runs as if the system doesn't exist.

The core tenets of this system are:

* **Contextual Intelligence**: Automatically track nested scopes and enrich metrics with context.
* **Extensibility via Protocols**: Allow users to integrate their own monitoring tools without rigid inheritance.
* **Zero-Overhead Guarantee**: Ensure that when profiling is disabled, the performance impact is not just minimal, but nonexistent.

-----

## 2. Architectural Design

The architecture is composed of `TelemetryContext` (the orchestrator) and the `TelemetryReporter` protocol (the extensible reporting mechanism).

### 2.1. The `TelemetryContext`: Central Orchestrator

The main interface for all instrumentation is the `TelemetryContext`. A developer interacts with a single `tele` object, which manages the scope stack, timing, and dispatching of metrics to all registered reporters.

Its primary interface is the `scope` context manager:

```python
# The primary interaction pattern
with self.tele.scope("batch.total_processing", question_count=len(questions)):
    # ... code to be timed ...
    self.tele.gauge("token_efficiency", some_value)
```

This automatically handles timing, duration calculation, and hierarchical scope naming (e.g., `batch.total_processing.attempt`), and enriches data with metadata like call depth and parent scope.

### 2.2. The `TelemetryReporter` Protocol: "Good Citizen" Extensibility

We use a `typing.Protocol` instead of a traditional abstract base class. This allows any object that "looks like" a reporter (i.e., has the right methods) to be used, decoupling the library from user-specific implementations.

```python
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

## 3. The Zero-Overhead Guarantee

The system provides two layers of "zero-overhead" guarantees.

### 3.1. The Runtime Check

The `TelemetryContext`'s `enabled` property is an ultra-fast check. All methods begin with an `if not self.enabled: return` guard, which is extremely fast.

```python
@property
def enabled(self) -> bool:
    """Ultra-fast enabled check - single tuple boolean evaluation"""
    return bool(self.reporters)

def metric(self, name: str, value: Any, **metadata):
    if not self.enabled:
        return
    # ... logic ...
```

### 3.2. The Compile-Time Optimization Pattern

For the most performance-critical code paths, a module-level constant (`_TELEMETRY_ENABLED`) completely removes the instrumentation at import time.

```python
# At module level - evaluated once at import time
_TELEMETRY_ENABLED = (
    os.getenv("GEMINI_TELEMETRY") == "1" or os.getenv("DEBUG") == "1"
)

if _TELEMETRY_ENABLED:
    # Full implementation
    def tele_scope(ctx, name, **metadata):
        return ctx.scope(name, **metadata)
else:
    # Zero-overhead no-op that does nothing
    @contextmanager
    def tele_scope(ctx, name, **metadata):
        yield ctx

# --- Usage in performance-critical code ---
# When disabled, this becomes just the inner block with no function call
with tele_scope(self.tele, "critical_operation"):
    expensive_operation()
```

When `_TELEMETRY_ENABLED` is `False`, the module-level `tele_scope` function becomes a no-op context manager, ensuring **zero** performance cost.

-----

## 4. Internal Tooling: The `SimpleReporter`

The library includes `SimpleReporter`, an in-memory reporter for internal development and debugging. **It is not intended as a primary user-facing feature.** Its purpose is to provide a quick way to inspect telemetry during development of the `gemini-batch` library itself.

### How to Use `SimpleReporter` for Debugging

The `SimpleReporter` is automatically enabled when the `GEMINI_TELEMETRY=1` or `DEBUG=1` environment variable is set. To use it, you must manually instantiate it and retrieve its report.

1. **Set the environment variable**:

    ```bash
    export GEMINI_TELEMETRY=1
    ```

2. **Instantiate and inject it**:
    In your test or script, create a `SimpleReporter` and `TelemetryContext`.

    ```python
    from gemini_batch.telemetry import SimpleReporter, TelemetryContext
    from gemini_batch import BatchProcessor

    # Create the reporter and context
    simple_reporter = SimpleReporter()
    tele_context = TelemetryContext(simple_reporter)

    # Inject it into the processor
    processor = BatchProcessor(telemetry_context=tele_context)

    # ... run your operations ...
    processor.process_questions(...)

    # Print the report at the end
    print(simple_reporter.get_report())
    ```

This explicit, manual process makes it clear that `SimpleReporter` is a developer tool, not a "plug-and-play" feature for end-users. Users wanting monitoring capabilities should be directed to `docs/TELEMETRY.md` and the corresponding example.
