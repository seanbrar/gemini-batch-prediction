# Documentation Authoring Templates

This page orients contributors to our documentation templates and where to find them. The full, detailed templates are maintained internally to keep user-facing docs focused.

- Audience: documentation contributors.
- Use with the following contributor resources:
  - Documentation Style Guide: [docs_style_guide.md](docs_style_guide.md)
  - Docs Quality Checklist: [docs_quality_checklist.md](docs_quality_checklist.md)

Where to find the full templates:

- Internal authoring templates live at: `dev/internal_only/guidance/authoring_templates.md` (in the repository). Use those copy-ready blocks when drafting pages.

Safety reminder for all docs:

!!! warning "Safety"

- Never include secrets (API keys, tokens). Use placeholders and `.env`.
- Call out cost/tier and rate-limit impacts near commands and scripts.
- Note telemetry/privacy flags where relevant.

Notes for Reference pages:

- API reference content is sourced via mkdocstrings; do not hand-edit auto-generated sections. Keep docstrings accurate and complete (Google style, typed).

Last reviewed: 2025‑09
