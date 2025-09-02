# Gemini Batch Documentation

Welcome. This site follows the Diátaxis framework:

- Tutorials: Step-by-step introductions to get you productive quickly.
- How-to guides: Task-oriented recipes for common developer workflows.
- Explanation: Architecture, concepts, decisions, and design rationale.
- Reference: Factual, API-like documentation and operational details.

Note: Some Explanation pages describe a target architecture (Command Pipeline). Tutorials, How-to guides, and Reference reflect the current API.

Start here:

- Tutorials → Quickstart: install, first run, success checks
- Tutorials → First Batch: multiple prompts and sources
- How‑to → Installation: Releases or source, environment setup
- How‑to → FAQ and Troubleshooting: common first‑run issues
- Reference → CLI (`gb-config`): check readiness with `doctor`
- Explanation → Architecture at a Glance
- Reference → API overview; How‑to → Configuration

## Onboarding checklist

- Install from Releases or source and verify `import gemini_batch`.
- Run the Quickstart hello example (mock mode) and confirm expected output.
- When ready, set `GEMINI_API_KEY`, `GEMINI_BATCH_TIER`, and `GEMINI_BATCH_USE_REAL_API=1`.
- Run `gb-config doctor` until no issues are reported.
- Proceed to `run_batch` examples in the Cookbook or Tutorials.
