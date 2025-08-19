"""CLI entry point for configuration introspection.

Usage:
    python -m gemini_batch.config
    python -m gemini_batch.config --check
    python -m gemini_batch.config --json
"""

from .introspection import main

if __name__ == "__main__":
    main()
