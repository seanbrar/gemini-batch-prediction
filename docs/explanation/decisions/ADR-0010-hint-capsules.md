# ADR‑0010 — Hint Capsules (Extensions → Core)

**Status:** Accepted (minimal pass)
**Date:** 2025‑08‑23
**Scope:** `planner`, `api_handler`, `result_builder`; optional types in `pipeline/hints.py`; `InitialCommand.hints`

## Context

We need a provider‑neutral way for extensions to express advanced intent to the pipeline (cache reuse identity, conservative token bounds, extraction preferences) without:

* introducing provider branches into core,
* changing handler responsibilities, or
* relying on implicit registry side‑effects.

At the same time, we must preserve **radical simplicity**: *less surface, more leverage*.

## Decision

Adopt **Hint Capsules** as a small union of frozen dataclasses that may be carried on `InitialCommand.hints`. Handlers may *optionally* read relevant hints:

* **Planner** reads `CacheHint` and `EstimationOverrideHint` to set `ExplicitCachePlan` and adjust `TokenEstimate.max_tokens` (pure transform).
* **API Handler** performs a best‑effort read of `ExecutionCacheName` and may treat it as an explicit cache intent for resilience: on a cache‑related failure it retries once without cache (same behavior as when an explicit cache plan was present). No other resilience semantics change.
* **Result Builder** optionally biases transform order with `ResultHint` while preserving Tier‑2 fallback.

Unknown hints are ignored. No handler gains provider‑specific code.

## Consequences

### Positive

* Single, explicit seam for extension intent; improved auditability and testability.
* No new control flow; handlers still do one thing each.
* Conservative, fail‑soft behavior: system remains correct even if hints are missing or partially supported.
* Observability: handlers may emit telemetry about hint usage for audit purposes.

### Negative / Risks

* Heterogeneous provider support means some hints become no‑ops.
* Temptation to add too many hint types. Mitigation: keep the union small and neutral; push provider details behind adapters.

## Alternatives considered

1. **Registry‑only signaling**

   * *Pros*: zero API change.
   * *Cons*: implicit, hard to audit, limited beyond caching.

2. **PlanDirectives DSL**

   * *Pros*: very explicit, audit‑friendly.
   * *Cons*: overkill for the goal; increases surface and maintenance.

3. **Executor‑level transient bag**

   * *Pros*: no type change on `InitialCommand`.
   * *Cons*: action‑at‑a‑distance risk; weaker audit trail unless duplicated into telemetry.

## Detailed design (minimal pass)

* Add `InitialCommand.hints: tuple[Hint, ...] | None = None`.
* Planner: read `CacheHint`, `EstimationOverrideHint` and produce the same `ExecutionPlan` shape with optional `ExplicitCachePlan` and adjusted estimate. No new branches beyond the data transform.
* API Handler: unchanged for minimal pass; optional best‑effort override of cache name is permitted without altering resilience semantics.
* Result Builder: add `_sorted_transforms_for(hints)` which, when `ResultHint(prefer_json_array=True)` is present, bubbles a `json_array` transform before the usual priority sort.

## Invariants

* `hints is None` ⇒ identical behavior and outputs.
* Planner remains the **only** place that computes token estimates.
* API Handler remains the **only** place that talks to adapters; the only change to retries is a single no‑cache retry when an exec‑time cache override is provided (same as with an explicit cache plan).
* Result Builder always returns success via Tier‑2 fallback.

## Migration & rollout

1. Introduce the hint types and `InitialCommand.hints` with handler reads present but behavior identical when hints are absent.
2. Update the conversation extension to map `ConversationState.cache` to `CacheHint` (and later opt‑in to `ResultHint` if desired).
3. Add a small test matrix covering: no‑op behavior, deterministic cache planning, reuse‑only policy, and result biasing safety.

Rollback: removing hint reads leaves the pipeline identical to pre‑hint behavior; data field can remain unused.

## Open questions / follow‑ups

* Add `BatchHint` once vectorized adapters surface per‑prompt outputs/metrics consistently.
* Consider `ExecAdapterHint` when providers expose stable, documented keys; keep them namespaced.
* Telemetry integration: handlers emit minimal hint usage metrics for audit purposes.

## References

* Command Pipeline & Prompting docs
* ADR‑0008 Conversation Extension
* RFP‑0001 Vectorization and Fan‑out
