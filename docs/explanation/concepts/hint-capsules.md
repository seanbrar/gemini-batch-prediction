# Hint Capsules – Conceptual Overview

> **Status:** Accepted (minimal pass)
> **Audience:** Contributors and advanced users
> **Position:** Explanation (what/why). Not an API reference or how‑to.
> **Scope:** Planner/API Handler/Result Builder touchpoints enabling *extension‑provided* hints in a provider‑neutral, fail‑soft way.

---

## Purpose

**Hint Capsules** are a tiny, immutable way for *extensions* to express intent to the core pipeline without coupling the core to any domain (e.g., “conversation”). They strengthen **radical simplicity** by keeping the control flow unchanged while allowing advanced behaviors to be **data‑driven**.

The first release supports these hints:

* `CacheHint` — deterministic cache identity and policy knobs.
* `EstimationOverrideHint` — conservative adjustments to token estimates (planner‑scoped, no provider coupling).
* `ResultHint` — non‑breaking transform preferences for extraction.
* `ExecutionCacheName` — best‑effort execution‑time cache name override (API handler reads it and, on cache‑related failure, performs a single no‑cache retry mirroring explicit cache plan semantics).

> Future (non‑breaking) additions may include `BatchHint` and `ExecAdapterHint` once adapters/telemetry warrant them.

---

## Why this exists (and what it replaces)

Without a neutral hint seam, extensions either:

* overreach into handler internals, or
* rely on opaque registries/side‑effects, or
* fork provider adapters.

**Hint Capsules** provide one *explicit*, *typed* place to express: “I’d prefer to reuse this cache identity,” “widen max tokens conservatively,” or “prefer JSON array extraction,” while preserving the Command Pipeline’s invariants and **single SDK seam**.

---

## Design tenets (Rubric alignment)

* **Radical Simplicity:** one optional field `InitialCommand.hints: tuple[Hint, ...] | None`; no new handlers, no new control branches.
* **Data over Control:** hints are *data*, consumed by existing stages via small, pure transforms.
* **Explicit over Implicit:** unknown hints are ignored; no hidden globals or ambient state.
* **Immutability:** hints travel with the immutable command; handlers remain stateless.
* **Single Provider Seam:** provider‑specific details remain inside adapters; core surfaces only neutral shapes.
* **Audit‑grade:** telemetry may record “hints seen,” but behavior never *depends* on undeclared state.

---

## Conceptual model

```text
InitialCommand(hints?) ──► Execution Planner ──► API Handler ──► Result Builder
         │                    │ (cache key,         │ (best‑effort        │ (transform
         │                    │ estimation range)   │ cache override)     │ preference)
         ▼                    ▼                     ▼                     ▼
     Fail‑soft           Plan remains pure;   Provider seam intact;   Tier‑1 bias only;
  (hints optional)       explicit cache plan  retries/fallback same   Tier‑2 fallback unchanged
```

### Hints (minimal pass)

* **`CacheHint`**

  * Fields: `deterministic_key: str`, `artifacts: tuple[str, ...] = ()`, `ttl_seconds: int | None = None`, `reuse_only: bool = False`.
  * **Planner:** Sets/overrides `ExplicitCachePlan` (`deterministic_key`, `ttl_seconds`, and create policy via `reuse_only`).
  * **API Handler:** unchanged (caching still routed through plan + capabilities). When a cache is created or reused, implementations may record hint-provided `artifacts` for audit purposes.

* **`EstimationOverrideHint`**

  * Fields: `widen_max_factor: float = 1.0`, `clamp_max_tokens: int | None = None`.
  * **Planner:** Applies a *pure* transform to `TokenEstimate.max_tokens` (widen then clamp), maintaining planner ownership of estimation logic. No runtime/provider coupling.

* **`ResultHint`**

  * Fields: `prefer_json_array: bool = False`.
  * **Result Builder:** Optionally biases Tier‑1 transform order (e.g., bubble `json_array`); **Tier‑2 minimal projection** guarantees success regardless.
  * When biasing is active, adds `prefer_json_array` flag to diagnostics and sets `metrics.hints.prefer_json_array = True` in the result envelope.

* **`ExecutionCacheName`**

  * Fields: `cache_name: str`.
  * **API Handler:** Best‑effort override of cache name at execution time. On cache‑related failure, triggers a single no‑cache retry (same resilience as explicit cache plans).

> No hint *requires* handler changes elsewhere; the control path and error semantics remain the same.

---

## Invariants & properties

* **I1 — No‑op by default:** `hints=None` yields identical behavior and outputs.
* **I2 — Planner owns estimation:** overrides are planner‑scoped transforms; API handler only validates/attaches usage telemetry.
* **I3 — Deterministic caching:** `CacheHint.deterministic_key` produces explicit cache plans; reuse‑only never hard‑fails planning.
* **I4 — Provider neutrality:** no provider branches in core; adapters may ignore namespaced details until supported (future additions only).
* **I5 — Guaranteed results:** Result Builder's Tier‑2 fallback keeps the system fail‑soft even with misleading or partial hints.
* **I6 — Observability:** Hints may generate telemetry data for audit and debugging purposes without affecting core behavior.

---

## Interaction with other concepts

* **Command Pipeline:** Hints *decorate* the initial command; all handler responsibilities remain intact.
* **Prompting System:** Unaffected. Prompt assembly precedes hint consumption; cache keys still include system text when present.
* **Vectorization & Fan‑out (RFP‑0001):** `BatchHint` is explicitly out of scope for the minimal pass; the seam is compatible when added.

---

## Rationale & trade‑offs

* **Why a single hints field?** It centralizes advanced intent in one predictable place, improving discoverability and testability.
* **Why fail‑soft?** Extensions evolve faster than core; ignoring unknown hints avoids coupling and preserves stability.
* **Why not registry‑only?** Registries alone are implicit. Hints keep decisions explicit while still allowing registries to assist (e.g., cache name reuse).
* **Why not a DSL?** Too heavy for the goal; Hint Capsules are a *toe‑hold* for power without extra concepts.

---

## Minimal type sketch (illustrative)

```py
@dataclass(frozen=True)
class CacheHint:  # planner‑scoped
    deterministic_key: str
    artifacts: tuple[str, ...] = ()
    ttl_seconds: int | None = None
    reuse_only: bool = False

@dataclass(frozen=True)
class EstimationOverrideHint:  # planner‑scoped
    widen_max_factor: float = 1.0
    clamp_max_tokens: int | None = None

@dataclass(frozen=True)
class ResultHint:  # result‑scoped
    prefer_json_array: bool = False
```

> Real API signatures live in the code; this is not a reference spec.

---

## Examples (high‑level intent)

* **Conversation cache identity**: Extension maps `ConversationState.cache.key` → `CacheHint.deterministic_key` so the planner emits an explicit cache plan; providers that support explicit caching reuse it deterministically.
* **Tight cost guardrails**: An evaluation tool sets `EstimationOverrideHint(widen_max_factor=1.1, clamp_max_tokens=16000)` for safer rate‑limit planning.
* **JSON‑first extraction**: A structured data workload sets `ResultHint(prefer_json_array=True)` to bias Tier‑1 extraction; fallback keeps results stable if the model outputs text.

---

## Risks & mitigations

* **Over‑hinting:** treat hints as *preferences*, not guarantees. Mitigation: document fail‑soft semantics and keep planner policies conservative.
* **Adapter heterogeneity:** not all providers support explicit caching/telemetry. Mitigation: hints are optional; registries and fallback paths remain.
* **Surface creep:** confine new capability hints to the same small union; avoid growing handler responsibilities.

---

## Related documents

* Command Pipeline – Conceptual Overview (`docs/explanation/concepts/command-pipeline.md`)
* Prompting System – Conceptual Overview (`docs/explanation/concepts/prompting.md`)
* ADR‑0001 Command Pipeline (`docs/explanation/decisions/ADR-0001-command-pipeline.md`)
* ADR‑0008 Conversation Extension (`docs/explanation/decisions/ADR-0008-conversation.md`)
* RFP‑0001 Batch Vectorization and Fan‑out (`docs/explanation/decisions/RFP-0001-batch-vectorization-and-fanout.md`)
