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
- Demo data (on demand):
  - `make demo-data` seeds text-medium and a minimal multimodal pack by default.
  - Customize with `TEXT=full` or `MEDIA=none` if needed.
  - Recipes accept `--input` (file or directory). If omitted, they look for `cookbook/data/demo/text-medium/` and print a friendly hint if missing.
- BYOF: pass `--input path/to/your/file_or_dir` to use your own files.
- Heads up: larger files and more files will increase runtime and token usage (cost). Keep inputs small for quick demos.
- Add `--log-cli-level=INFO` to pytest commands to view more logs.

---

## 🧭 Quick Run Checklist

- Install deps: `make install-dev`
- Seed demo data: `make demo-data`
- Export env (example):
  - `export GEMINI_API_KEY=...`
  - `export GEMINI_BATCH_TIER=tier_1` (match your billing)
  - `export GEMINI_BATCH_USE_REAL_API=true`
- Run from repo root with `PYTHONPATH='.'` to resolve `cookbook.*` imports, e.g.:
  - `PYTHONPATH='.' python cookbook/production/resume-on-failure.py`

Why `PYTHONPATH='.'`? When executing a file directly (e.g., `python cookbook/...py`),
`sys.path[0]` is the script directory, not the repo root. Adding the repo root to
`PYTHONPATH` ensures `from cookbook...` imports work reliably.

## ▶️ Example Commands

- Production resume: persists per-item status and manifest
  - `PYTHONPATH='.' python cookbook/production/resume-on-failure.py`
  - Outputs: `outputs/manifest.json`, per-item JSON under `outputs/items/`

- Context caching (explicit create → reuse)
  - `PYTHONPATH='.' python cookbook/optimization/context-caching-explicit.py --limit 2`
  - Shows warm vs reuse token totals and effective cache hits for the batch

- Cache warming with TTL and deterministic key
  - `PYTHONPATH='.' python cookbook/optimization/cache-warming-and-ttl.py --limit 2`
  - Prints warm vs reuse tokens and cache hits (warm→reuse)

Tips:

- Use `--input path/to/dir_or_file` to analyze your own content.
- Use `--limit N` on caching recipes to bound file count and speed up runs.

## 🧠 Caching Recipe Notes

- Token savings vs “hits”: Providers may count tokens differently on warm vs reuse.
  Always check both the token totals and the reported cache hits in metrics.
- The caching recipes reuse a single executor under the hood so the in-memory
  cache registry persists between warm and reuse runs.
- Effective hit reporting: recipes compute hits from available per-call metadata;
  some providers may not expose all counters. Treat token deltas as indicative, not absolute.

## 🛠️ Troubleshooting

- ModuleNotFoundError: `cookbook`
  - Add repo root to Python path: `PYTHONPATH='.' python cookbook/...`
- No demo data found
  - Run `make demo-data` (or pass `--input your_dir`)
- 429 / throttling
  - Set `GEMINI_BATCH_TIER` to match your billing tier; reduce `--limit` or input size.
- Slow runs / high tokens
  - Keep to small files for demos; use `--limit`, and prefer text over large PDFs/videos.

## 🔐 Secrets & Config

- Store secrets in `.env` (gitignored): `GEMINI_API_KEY=...`
- Optional envs:
  - `GEMINI_BATCH_TIER` (e.g., `tier_1`)
  - `GEMINI_BATCH_USE_REAL_API` (`true` to call the provider)
  - Model override via `pyproject.toml` or env per docs
