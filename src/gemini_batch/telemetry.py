from collections import deque  # noqa: D100
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, List, Protocol, TypeAlias, Union, runtime_checkable  # noqa: UP035

from typing_extensions import deprecated  # noqa: UP035

log = logging.getLogger(__name__)

# Context-aware state for thread/async safety
_scope_stack_var: ContextVar[List[str]] = ContextVar("scope_stack", default=None)  # noqa: UP006
_call_count_var: ContextVar[int] = ContextVar("call_count", default=0)

# Minimal-overhead optimization - evaluated once at import time
_TELEMETRY_ENABLED = os.getenv("GEMINI_TELEMETRY") == "1" or os.getenv("DEBUG") == "1"


@runtime_checkable
class TelemetryReporter(Protocol):
    """Duck-typed protocol for telemetry reporters."""

    def record_timing(self, scope: str, duration: float, **metadata) -> None: ...  # noqa: ANN003, D102
    def record_metric(self, scope: str, value: Any, **metadata) -> None: ...  # noqa: ANN003, ANN401, D102


@dataclass(frozen=True, slots=True)
class _NoOpTelemetryContext:
    """An immutable and stateless no-op context, optimized for negligible overhead."""

    def __call__(self, name: str, **metadata):  # noqa: ANN003, ANN204, ARG002
        return self  # Self is already a context manager

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001, ANN204
        return None

    def metric(self, name: str, value: Any, **metadata):  # noqa: ANN003, ANN202, ANN401
        pass

    def time(self, name: str, **metadata):  # noqa: ANN003, ANN202, ARG002
        return self

    def count(self, name: str, increment: int = 1, **metadata):  # noqa: ANN003, ANN202
        pass

    def gauge(self, name: str, value: float, **metadata):  # noqa: ANN003, ANN202
        pass


class _EnabledTelemetryContext:
    """Full-featured telemetry context when enabled."""

    __slots__ = ("reporters",)

    def __init__(self, *reporters: TelemetryReporter):  # noqa: ANN204
        self.reporters = reporters

    def __call__(self, name: str, **metadata):  # noqa: ANN003, ANN204
        return self._create_scope(name, **metadata)

    @contextmanager
    def _create_scope(self, name: str, **metadata):  # noqa: ANN003, ANN202
        """The actual scope implementation when enabled."""
        if not name or not isinstance(name, str):
            raise ValueError("Scope name must be a non-empty string")  # noqa: EM101, TRY003

        scope_stack = _scope_stack_var.get() or []
        call_count = _call_count_var.get()
        scope_path = ".".join(scope_stack + [name])  # noqa: RUF005
        start_time = time.perf_counter()

        scope_token = _scope_stack_var.set(scope_stack + [name])  # noqa: RUF005
        count_token = _call_count_var.set(call_count + 1)

        try:
            yield self  # Return self so chained methods work
        finally:
            duration = time.perf_counter() - start_time
            _scope_stack_var.reset(scope_token)
            _call_count_var.reset(count_token)

            final_stack = _scope_stack_var.get() or []
            final_count = _call_count_var.get()

            enhanced_metadata = {
                "depth": len(final_stack),
                "call_count": final_count,
                "parent_scope": ".".join(final_stack) if final_stack else None,
                **metadata,
            }

            for reporter in self.reporters:
                try:
                    reporter.record_timing(scope_path, duration, **enhanced_metadata)
                except Exception as e:
                    log.error(  # noqa: G201
                        "Telemetry reporter '%s' failed: %s",
                        type(reporter).__name__,
                        e,
                        exc_info=True,
                    )

    def metric(self, name: str, value: Any, **metadata):  # noqa: ANN003, ANN202, ANN401
        """Record a metric within current scope context"""  # noqa: D415
        scope_stack = _scope_stack_var.get() or []
        scope_path = ".".join(scope_stack + [name])  # noqa: RUF005
        enhanced_metadata = {
            "depth": len(scope_stack),
            "parent_scope": ".".join(scope_stack) if scope_stack else None,
            **metadata,
        }
        for reporter in self.reporters:
            try:
                reporter.record_metric(scope_path, value, **enhanced_metadata)
            except Exception as e:
                log.error(  # noqa: G201
                    "Telemetry reporter '%s' failed: %s",
                    type(reporter).__name__,
                    e,
                    exc_info=True,
                )

    # Convenience methods for chaining
    def time(self, name: str, **metadata):  # noqa: ANN003, ANN202
        """Alias for scope() - more intuitive for timing operations"""  # noqa: D415
        return self(name, **metadata)

    def count(self, name: str, increment: int = 1, **metadata):  # noqa: ANN003, ANN202
        """Record a counter metric"""  # noqa: D415
        self.metric(name, increment, metric_type="counter", **metadata)

    def gauge(self, name: str, value: float, **metadata):  # noqa: ANN003, ANN202
        """Record a gauge metric"""  # noqa: D415
        self.metric(name, value, metric_type="gauge", **metadata)


_NO_OP_SINGLETON = _NoOpTelemetryContext()

TelemetryContextProtocol: TypeAlias = Union[  # noqa: UP007, UP040
    _EnabledTelemetryContext, _NoOpTelemetryContext  # noqa: COM812
]


def TelemetryContext(*reporters: TelemetryReporter) -> TelemetryContextProtocol:  # noqa: N802
    """
    Factory that returns either a full-featured telemetry context or a
    single, shared, no-op instance for maximum performance when disabled.
    """  # noqa: D205, D212
    if _TELEMETRY_ENABLED and reporters:
        return _EnabledTelemetryContext(*reporters)
    else:  # noqa: RET505
        # Always return the same, pre-existing no-op instance.
        return _NO_OP_SINGLETON


@deprecated("Use ctx(name, **metadata) instead")
def tele_scope(ctx: TelemetryContextProtocol, name: str, **metadata):  # noqa: ANN003, ANN201, D103
    return ctx(name, **metadata)


class _SimpleReporter:
    """Built-in reporter for development use.

    This reporter collects metrics in memory. To view the collected data,
    call the `get_report()` method and print the result.
    """

    def __init__(self, max_entries_per_scope: int = 1000):  # noqa: ANN204
        self.max_entries = max_entries_per_scope
        self.timings = {}
        self.metrics = {}

    def record_timing(self, scope: str, duration: float, **metadata):  # noqa: ANN003, ANN202
        if scope not in self.timings:
            self.timings[scope] = deque(maxlen=self.max_entries)
        self.timings[scope].append((duration, metadata))

    def record_metric(self, scope: str, value: Any, **metadata):  # noqa: ANN003, ANN202, ANN401
        if scope not in self.metrics:
            self.metrics[scope] = deque(maxlen=self.max_entries)
        self.metrics[scope].append((value, metadata))

    def print_report(self):  # noqa: ANN202
        """Prints the report to stdout if any data was collected."""
        if self.timings or self.metrics:
            print(self.get_report())  # noqa: T201

    def get_report(self) -> str:
        """Generate hierarchical telemetry report"""  # noqa: D415
        lines = ["=== Telemetry Report ===\n"]

        # Group by hierarchy
        timing_tree = self._build_hierarchy(self.timings)
        self._format_tree(timing_tree, lines, "Timings")

        if self.metrics:
            lines.append("\n--- Metrics ---")
            for scope, values in sorted(self.metrics.items()):
                total = sum(v[0] for v in values if isinstance(v[0], (int, float)))
                lines.append(
                    f"{scope:<40} | Count: {len(values):<4} | Total: {total:,.0f}"  # noqa: COM812
                )

        return "\n".join(lines)

    def _build_hierarchy(self, data):  # noqa: ANN001, ANN202
        """Build tree structure from dot-separated scope names"""  # noqa: D415
        tree = {}
        for scope, values in data.items():
            parts = scope.split(".")
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = values
        return tree

    def _format_tree(self, tree, lines, title, depth=0):  # noqa: ANN001, ANN202
        """Format hierarchical tree with indentation"""  # noqa: D415
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
                    f"Total: {total_time:.4f}s"  # noqa: COM812
                )
