# Gemini Batch Architecture – At a Glance

> Status: Target architecture. Some components still map to legacy classes on this branch. The new token estimation system is implemented on a separate branch; see Related Docs.
>
> Audience: Contributors and advanced users.
> Prerequisites: Familiarity with the project structure and basic async Python.

## Purpose

Provide developer-consumers with a predictable, testable, and extensible way to run batched Gemini API calls at scale. The architecture is designed for clarity, composability, and performance.

---

## Key Principles

- **Radical Simplicity** – Even advanced features should feel obvious in retrospect.
- **Explicit over Implicit** – No hidden state or “magic”; data and control flow are transparent.
- **Data-Centricity** – Complex state lives in rich, immutable objects; transformations are simple and stateless.
- **Architectural Robustness** – Invalid states are structurally impossible.
- **Superior Developer Experience** – Components are isolated, testable, and easy to extend.

---

## Primary Pattern – Command Pipeline

The Command Pipeline is an **asynchronous, unidirectional series of stateless handlers** that transform a request from raw inputs into a final, enriched result.

**Flow:**

```text
Command → Source Handler → Execution Planner → API Handler → Result Builder → Result
```

---

## Core Components

- **Command** – Immutable description of the request (sources, prompts, configuration)
- **Source Handler** – Resolves and normalizes raw inputs
- **Execution Planner** – Makes strategic decisions (token budgeting, caching, batching, prompt assembly)
- **API Handler** – Executes the plan via the Gemini API, handling rate limits and errors
- **Result Builder** – Parses output, validates schema, merges telemetry, calculates metrics
- **Executor** – Orchestrates handlers in order

---

## Why This Architecture

Replaces earlier monolithic design (`BatchProcessor` + `GeminiClient`) that had:

- Hidden sequencing
- Mixed responsibilities
- Synchronous bottlenecks
- Hard-to-extend decision logic

The pipeline separates concerns, enforces state validity, and enables async-first scalability.

---

## Quality Attributes

- **Testability** – Handlers testable in isolation
- **Extensibility** – Add steps without touching existing ones
- **Robustness** – Type system enforces valid transitions
- **Transparency** – Pipeline structure is clear and predictable

---

## Diagram

```mermaid
flowchart LR
    User[Developer App] --> Command
    Command --> SH[Source Handler]
    SH --> EP[Execution Planner]
    EP --> AH[API Handler]
    AH --> RB[Result Builder]
    RB --> Result
```

### Handler roles at a glance

- Source Handler: Normalize raw inputs into structured `Source` data.
- Execution Planner: Decide batching, caching, token budgeting, and prompt assembly.
- API Handler: Execute the plan (rate limits, retries, provider SDK calls).
- Result Builder: Parse output, validate schemas, and merge telemetry/metrics.

---

## Related Docs

- [Concept – Command Pipeline](./concepts/command-pipeline.md)
- [Deep Dive – Command Pipeline Spec](./deep-dives/command-pipeline-spec.md)
- [ADR-0001 – Command Pipeline](./decisions/ADR-0001-command-pipeline.md)
- [Architecture Rubric](./architecture-rubric.md)
- [Concept – Token Counting & Estimation](./concepts/token-counting.md)
- [Glossary](./glossary.md)
