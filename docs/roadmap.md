# Project Roadmap

A simple view of what’s coming and how it will help. Timelines are directional and may change.

## Coming Soon

- Upload stability for media: waits for provider uploads to be ready before use. Effect: fewer transient errors in video/image batches.
- Structured outputs: optional JSON schema guidance to return cleaner JSON. Effect: less post‑processing in your pipelines.

## Next Up

- Smarter retries: jittered backoff on transient API errors. Effect: higher batch completion without manual restarts.
- Clearer cache metrics: distinguish intended vs. effective cache use. Effect: easier to reason about hits/misses in dashboards.

## On the Horizon

- Batch fanout research: improved parallelization for very large jobs. Effect: faster end‑to‑end runs when scaling up.
- Response format options: choose output format per call or pipeline. Effect: simpler integration with downstream tools.
