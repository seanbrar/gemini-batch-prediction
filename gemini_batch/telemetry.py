from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
import time
from typing import Any, List, Protocol

log = logging.getLogger(__name__)

# Context-aware state for thread/async safety
_scope_stack_var: ContextVar[List[str]] = ContextVar("scope_stack", default=None)
_call_count_var: ContextVar[int] = ContextVar("call_count", default=0)

# Zero-overhead compile-time optimization - evaluated once at import time
_TELEMETRY_ENABLED = os.getenv("GEMINI_TELEMETRY") == "1" or os.getenv("DEBUG") == "1"

if _TELEMETRY_ENABLED:
    # Full implementation for performance-critical code
    def tele_scope(ctx, name, **metadata):
        return ctx.scope(name, **metadata)
else:
    # Zero-overhead no-op that compiles to just the inner block
    @contextmanager
    def tele_scope(ctx, name, **metadata):
        yield ctx


class TelemetryReporter(Protocol):
    """
    Duck-typed protocol for telemetry reporters - no inheritance required!
    Users can implement any subset of these methods.
    """

    def record_timing(self, scope: str, duration: float, **metadata) -> None: ...
    def record_metric(self, scope: str, value: Any, **metadata) -> None: ...


class TelemetryContext:
    """
    Zero-overhead contextual telemetry with extensible reporting.

    Provides hierarchical scope tracking and metrics collection with optional
    reporters. When disabled (no reporters), operations have zero runtime cost.

    Thread-safe and async-ready via contextvars.

    Example:
        >>> reporter = SimpleReporter()
        >>> tele_context = TelemetryContext(reporter)
        >>> with tele_context.scope("operation"):
        ...     expensive_operation()
        >>> print(reporter.get_report())
    """

    def __init__(self, *reporters: TelemetryReporter):
        self.reporters = reporters

    @property
    def enabled(self) -> bool:
        """Ultra-fast enabled check - single tuple boolean evaluation"""
        return bool(self.reporters)

    @contextmanager
    def scope(self, name: str, **metadata):
        """Contextual scope that automatically tracks nesting and relative timings"""
        # Validate scope name
        if not name or not isinstance(name, str):
            raise ValueError("Scope name must be a non-empty string")

        if not self.enabled:
            yield self
            return

        # Get thread-local state
        scope_stack = _scope_stack_var.get() or []
        call_count = _call_count_var.get()

        # Build hierarchical scope name
        scope_path = ".".join(scope_stack + [name])
        start_time = time.perf_counter()

        # Push scope and update count
        token = _scope_stack_var.set(scope_stack + [name])
        count_token = _call_count_var.set(call_count + 1)

        try:
            yield self  # Return self for chaining operations
        finally:
            # Pop scope and record
            duration = time.perf_counter() - start_time
            _scope_stack_var.reset(token)
            _call_count_var.reset(count_token)

            # Get latest state for metadata
            final_stack = _scope_stack_var.get() or []
            final_count = _call_count_var.get()

            # Enhanced metadata with context
            enhanced_metadata = {
                "depth": len(final_stack),
                "call_count": final_count,
                "parent_scope": ".".join(final_stack) if final_stack else None,
                **metadata,
            }

            # Report to all registered reporters safely
            for reporter in self.reporters:
                try:
                    reporter.record_timing(scope_path, duration, **enhanced_metadata)
                except Exception as e:
                    log.error(
                        "Telemetry reporter '%s' failed during record_timing: %s",
                        type(reporter).__name__,
                        e,
                        exc_info=True,
                    )

    def metric(self, name: str, value: Any, **metadata):
        """Record a metric within current scope context"""
        if not self.enabled:
            return

        scope_stack = _scope_stack_var.get() or []
        scope_path = ".".join(scope_stack + [name])
        enhanced_metadata = {
            "depth": len(scope_stack),
            "parent_scope": ".".join(scope_stack) if scope_stack else None,
            **metadata,
        }

        for reporter in self.reporters:
            try:
                reporter.record_metric(scope_path, value, **enhanced_metadata)
            except Exception as e:
                log.error(
                    "Telemetry reporter '%s' failed during record_metric: %s",
                    type(reporter).__name__,
                    e,
                    exc_info=True,
                )

    # Convenience methods for chaining
    def time(self, name: str, **metadata):
        """Alias for scope() - more intuitive for timing operations"""
        return self.scope(name, **metadata)

    def count(self, name: str, increment: int = 1, **metadata):
        """Record a counter metric"""
        self.metric(name, increment, metric_type="counter", **metadata)

    def gauge(self, name: str, value: float, **metadata):
        """Record a gauge metric"""
        self.metric(name, value, metric_type="gauge", **metadata)


class SimpleReporter:
    """Built-in reporter for development use.

    This reporter collects metrics in memory. To view the collected data,
    call the `get_report()` method and print the result.
    """

    def __init__(self, max_entries_per_scope: int = 1000):
        self.max_entries = max_entries_per_scope
        self.timings = {}
        self.metrics = {}

    def record_timing(self, scope: str, duration: float, **metadata):
        if scope not in self.timings:
            self.timings[scope] = deque(maxlen=self.max_entries)
        self.timings[scope].append((duration, metadata))

    def record_metric(self, scope: str, value: Any, **metadata):
        if scope not in self.metrics:
            self.metrics[scope] = deque(maxlen=self.max_entries)
        self.metrics[scope].append((value, metadata))

    def print_report(self):
        """Prints the report to stdout if any data was collected."""
        if self.timings or self.metrics:
            print(self.get_report())

    def get_report(self) -> str:
        """Generate hierarchical telemetry report"""
        lines = ["=== Telemetry Report ===\n"]

        # Group by hierarchy
        timing_tree = self._build_hierarchy(self.timings)
        self._format_tree(timing_tree, lines, "Timings")

        if self.metrics:
            lines.append("\n--- Metrics ---")
            for scope, values in sorted(self.metrics.items()):
                total = sum(v[0] for v in values if isinstance(v[0], (int, float)))
                lines.append(
                    f"{scope:<40} | Count: {len(values):<4} | Total: {total:,.0f}"
                )

        return "\n".join(lines)

    def _build_hierarchy(self, data):
        """Build tree structure from dot-separated scope names"""
        tree = {}
        for scope, values in data.items():
            parts = scope.split(".")
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = values
        return tree

    def _format_tree(self, tree, lines, title, depth=0):
        """Format hierarchical tree with indentation"""
        if depth == 0:
            lines.append(f"\n--- {title} ---")

        for key, value in sorted(tree.items()):
            indent = "  " * depth
            if isinstance(value, dict):
                lines.append(f"{indent}{key}:")
                self._format_tree(value, lines, title, depth + 1)
            else:
                # Leaf node - actual timing data
                durations = [v[0] for v in value]
                avg_time = sum(durations) / len(durations)
                total_time = sum(durations)
                lines.append(
                    f"{indent}{key:<30} | "
                    f"Calls: {len(durations):<4} | "
                    f"Avg: {avg_time:.4f}s | "
                    f"Total: {total_time:.4f}s"
                )
