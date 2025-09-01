# Gemini Batch Processing Cookbook

Problem-first recipes for real-world multimodal analysis

## 🚀 Getting Started

Perfect for first-time users

| Recipe | When you need to... | Difficulty | Time |
|--------|----------------------|------------|------|
| `getting-started/analyze-single-paper.py` | Extract key insights from one file | ⭐ | 5 min |
| `getting-started/batch-process-files.py` | Process multiple documents efficiently | ⭐⭐ | 8 min |
| `getting-started/extract-video-insights.py` | Pull highlights from a video | ⭐⭐ | 8 min |
| `getting-started/token-estimate-preview.py` | Estimate tokens/cost before running | ⭐⭐ | 5–8 min |
| `getting-started/structured-json-robust.py` | Get JSON with schema + fallbacks | ⭐⭐ | 8–10 min |
| `getting-started/youtube-qa-timestamps.py` | Q&A on YouTube with timestamp links | ⭐⭐ | 8 min |
| `getting-started/conversation-follow-ups.py` | Persisted follow-ups via ConversationEngine | ⭐⭐ | 8–10 min |

## 📚 Research Workflows

Academic and educational scenarios

- `research-workflows/literature-synthesis.py` — Synthesize findings across many papers
- `research-workflows/comparative-analysis.py` — Compare two or more sources side‑by‑side
- `research-workflows/content-assessment.py` — Assess course/lecture materials for learning objectives
- `research-workflows/fact-table-extraction.py` — Extract normalized fact rows to JSONL/CSV
- `research-workflows/multi-video-batch.py` — Compare/summarize across up to 10 videos
- `research-workflows/system-instructions-with-research-helper.py` — Apply system instructions while benchmarking efficiency

## ⚙️ Optimization

Performance, scale, and cost efficiency

- `optimization/cache-warming-and-ttl.py` — Warm caches with deterministic keys and TTL
- `optimization/chunking-large-docs.py` — Chunk very large docs and merge answers
- `optimization/large-scale-batching.py` — Fan‑out over many sources with bounded concurrency
- `optimization/multi-format-pipeline.py` — Analyze mixed media (PDF, image, video) together
- `optimization/context-caching-explicit.py` — Explicit cache create/reuse and token savings
- `optimization/long-transcript-chunking.py` — Token-aware transcript chunking + stitching
- `optimization/efficiency-comparison.py` — Compare vectorized vs naive for N prompts

## 🏭 Production Patterns

Reliability and observability

- `production/monitoring-telemetry.py` — Inspect per‑stage timings and metrics
- `production/resume-on-failure.py` — Persist state and rerun only failed items
- `production/custom-integrations.py` — Attach a custom telemetry reporter
- `production/rate-limits-and-concurrency.py` — Tier config + bounded concurrency behavior

## 🧩 Templates

- `templates/recipe-template.py` — Boilerplate for a new recipe
- `templates/custom-schema-template.py` — Start here for schema‑first extraction

Notes

- Set `GEMINI_API_KEY` and any model/tier env as needed.
- Run `make fetch-cookbook-data` to download small public samples into `cookbook/data/public/`.
- Scripts default to `cookbook/data/public/` (fallback to `examples/test_data` if present).
- Add `--log-cli-level=INFO` to pytest commands to view more logs.
