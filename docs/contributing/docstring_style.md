# Docstring Style Guide

This project uses Google-style docstrings. Keep them concise and practical, but document notable, surprising, or important behavior. Aim for clarity, accuracy, and utility.

## General Rules

- Use Google style sections: Args, Returns, Raises, Example(s), Notes.
- Keep summaries short (one sentence), then add detail only when useful.
- Prefer precise nouns/verbs; avoid repetition of obvious information.
- Document invariants, edge cases, and non-standard behavior (even if verbose).
- Match naming and types to actual code signatures (type hints are authoritative).
- Public APIs must have docstrings. Internal helpers should be documented when nontrivial.

## Examples

Function:

```python
def fetch_items(ids: Sequence[str], *, timeout_s: float = 10.0) -> list[Item]:
    """Fetch items by id with a bounded timeout.

    Args:
        ids: Unique item identifiers.
        timeout_s: Per-request timeout in seconds.

    Returns:
        Fetched items in the same order as `ids`.

    Raises:
        ItemNotFoundError: When any id cannot be fetched.
        TimeoutError: When the request exceeds `timeout_s`.

    Notes:
        Enforces order preservation and fails fast on the first missing id.
    """
```

Class:

```python
class ResultBuilder:
    """Build `ResultEnvelope` objects from finalized commands.

    Applies Tier 1 transforms by priority and falls back to an infallible
    MinimalProjection, optionally collecting diagnostics.

    Attributes:
        transforms: Tier 1 transform specs.
        enable_diagnostics: Attach diagnostics when True.
    """
```

## When to Add Detail

- Invariants: final shapes, ordering guarantees, idempotency, retry semantics.
- Performance: concurrency bounds, rate limits, cache behavior, size caps.
- Safety/telemetry: best-effort behavior, when errors are suppressed vs raised.
- Configuration: environment flags, defaults, precedence rules.

## Formatting Tips

- Wrap code, paths, and literals in backticks.
- Use short examples; prefer references to full guides when lengthy.
- Avoid copying type information into prose when it’s already in signatures.

## References

- Convention: Google style
- Lint: pydocstyle via Ruff (`[tool.ruff.lint.pydocstyle]`)
