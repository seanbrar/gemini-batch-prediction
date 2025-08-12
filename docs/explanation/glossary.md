# Glossary

- Command: Immutable description of a request (sources, prompts, config). May have typed variants (Initial, Resolved, Planned).
- Source: Structured metadata for an input item (type, identifier, mime, size, content_loader).
- ExecutionPlan: Explicit instructions for API execution (model, parts, config, caching, optional fallback).
- Handler: Stateless component that transforms one Command state into the next.
- GeminiExecutor: Orchestrator that runs a Command through Handlers.
- APIHandler: Handler that executes provider SDK calls and records actual usage.
- Result Builder: Handler that parses outputs, validates schema, and merges telemetry/metrics.
- Result: Unified result type (Success/Failure) used for control flow.
- TokenEstimate: Range-based estimate (min/expected/max) with confidence and optional breakdown.
- Estimation Adapter: Provider-specific, pure estimator used by the planner; no SDK calls.
