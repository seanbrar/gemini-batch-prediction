# Project Roadmap

This roadmap captures near- and mid‑term enhancements for the Gemini Batch
Pipeline. Core systems are currently frozen; any change below is a proposal that
must be scheduled, reviewed, and implemented behind stable interfaces.

## Status Legend

- Planned: item accepted for a future release window
- Proposed: needs design/estimates and approval
- Research: exploratory; not yet ready for design

## 1) Provider‑Enforced Structured Outputs (Proposed)

Goal: Leverage Gemini response schemas to bias generation and return strongly
typed JSON or enums, while keeping the pipeline provider‑neutral and backwards
compatible.

Why: Our current JSON support is extraction‑based (Tier‑1 transform bias via
`prefer_json_array`). It parses JSON reliably but does not influence model
decoding. Provider‑enforced schemas maximize precision and reduce post‑processing.

### User Story

- As a user, I want to specify a JSON schema (or Pydantic model) and receive
  provider‑validated JSON without additional prompt tricks, with results carried
  in `structured_data` and mirrored as strings in `answers` when applicable.

### High‑Level Design

1) ExecutionOptions extension (provider‑neutral)
   - Add optional fields:
     - `response_mime_type: str | None` (e.g., `"application/json"`)
     - `response_schema: Any | None` (provider‑specific schema object or a
       simple neutral description; see open questions)
   - Keep these fields inert for providers that do not support schemas.

2) Planning: carry options into `APICall.api_config`
   - Planner reads `ExecutionOptions` and, when present, writes
     `{"response_mime_type": ..., "response_schema": ...}` into each
     `APICall.api_config` alongside `system_instruction`.
   - Ensure cache identity remains stable (schema does not affect cache key
     unless explicitly decided; see below).

3) Adapter: GoogleGenAIAdapter passthrough
   - Adapter already converts a dict to `GenerateContentConfig` by setattr.
   - Confirm SDK accepts `response_mime_type` and `response_schema` on the
     `config` object; otherwise fall back to passing the raw dict.
   - Continue to normalize the provider response to
     `{ "text": ..., "usage": ..., ... }` or candidates‑shape.

4) ResultBuilder behavior
   - If provider returns `.parsed` or a structured payload, surface it under
     `result["structured_data"]` unchanged when JSON compatible.
   - Preserve Tier‑1 transforms as fallback (e.g., if schema not honored or
     parsing suppressed), maintaining current reliability guarantees.
   - Diagnostics: note when schema was requested vs. applied.

5) Telemetry & Diagnostics
   - Record flags:
     - `schema_requested`, `schema_applied`, `response_mime_type`
   - Optionally capture a summary (e.g., item count, key presence ratios).

### Backward Compatibility

- No change to default behavior. Structured outputs engage only when options
  are present. Existing `prefer_json_array` continues to work.

### API Surfaces Affected

- Python:
  - `ExecutionOptions` (new fields)
  - `make_execution_options(...)` (kwargs for the new fields)
  - Planner: propagate to `APICall.api_config`
  - Adapter: ensure passthrough to SDK `GenerateContentConfig`
  - ResultBuilder: optional pass‑through of provider parsed payloads

### Caching Considerations

- Default: schema does not alter cache identity to keep reuse broad.
- Option: add a conservative toggle to include schema hash into the shared cache
  key to avoid cross‑schema collisions for specific workloads.

### Security Considerations

- Do not serialize secrets within schemas into logs/telemetry.
- Validate that schemas are JSON‑serializable or have stable repr hashing.

### Open Questions / Decisions Needed

- Representing `response_schema`: accept provider objects, Pydantic models, or a
  neutral JSON Schema dict? Proposal: support any; adapter inspects type and
  passes through when understood.
- Do we want a minimal, neutral validation pass when provider parsing fails?

### Milestones

1. Spike: cookbook demo via direct SDK (no core changes)
2. Options + planner propagation (feature‑flagged)
3. Adapter passthrough + conformance tests
4. ResultBuilder diagnostics and pass‑through
5. Docs and examples

---

## Parking Lot (Future)

- Response format negotiation per‑call vs. per‑pipeline
- Enum‑only constrained outputs for classifiers
- Cross‑provider schema normalization utilities
