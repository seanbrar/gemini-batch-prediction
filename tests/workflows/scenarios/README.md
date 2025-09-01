# Scenarios

Curated end-to-end scenarios that exercise realistic flows with explicit inputs and expected outputs.

Guidelines:

- Use the `@pytest.mark.scenario` marker on all scenario tests.
- Prefer fixtures in `tests/conftest.py` and add dedicated helpers here only when necessary.
- Keep runtime reasonable; avoid external API calls unless explicitly marked as `api`.
