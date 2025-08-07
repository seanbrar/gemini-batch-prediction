"""Configuration management for the Gemini Batch Pipeline.

This module provides ways to configure the pipeline behavior, supporting both
explicit configuration and ambient context resolution.
"""

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import os
from typing import TypedDict

from .core.models import APITier

# --- Type-Safe Configuration Definition ---


class GeminiConfig(TypedDict, total=False):
    """Configuration options for the Gemini pipeline.

    All parameters are optional when creating the dict, but the system ensures
    they are resolved to concrete values before the pipeline is executed.
    """

    api_key: str
    model: str
    tier: str | APITier
    enable_caching: bool


# --- Ambient Configuration Resolution ---

# The context variable holds the current configuration for the execution context.
_ambient_config_var: contextvars.ContextVar[GeminiConfig] = contextvars.ContextVar(
    "gemini_batch_config"
)


def get_ambient_config() -> GeminiConfig:
    """Resolves configuration from the current context or environment.

    This function gathers configuration when none is provided explicitly.
    It follows a clear precedence:
    1. Configuration set by `config_scope`.
    2. Environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`).
    3. Sensible defaults.

    Returns:
        A fully-resolved GeminiConfig dictionary.
    """
    try:
        # If inside a config_scope, this will return the scoped config.
        return _ambient_config_var.get()
    except LookupError:
        # If no context is set, resolve from the environment.
        # This is the default behavior.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # In the new architecture, a missing API key is a critical
            # configuration error that should fail early.
            raise ValueError(
                "GEMINI_API_KEY is not set in the environment and no explicit "
                "config was provided."
            ) from None

        return GeminiConfig(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )


@contextmanager
def config_scope(config: GeminiConfig) -> Generator[None]:
    """Temporarily use a different configuration.

    This context manager allows running specific operations with different
    settings without affecting the global environment. It's thread-safe
    and async-safe, making it ideal for testing or specific use cases.

    Example:
        with config_scope(GeminiConfig(model="gemini-2.5-pro")):
            # Code inside this block will use gemini-2.5-pro
            executor = create_executor()
            await executor.execute(...)
    """
    token = _ambient_config_var.set(config)
    try:
        yield
    finally:
        _ambient_config_var.reset(token)
