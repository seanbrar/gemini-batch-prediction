"""Configuration scoping for entry-time overrides.

This module provides the updated config_scope() system that works with ResolvedConfig
and only affects resolution at entry time, not during pipeline execution.
"""

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
from pathlib import Path
from typing import Any

# Removed resolve_config import to break circular dependency
from .types import ResolvedConfig

# Context variable for ambient configuration during resolution
_ambient_resolved_config: contextvars.ContextVar[ResolvedConfig] = (
    contextvars.ContextVar("gemini_batch_resolved_config")
)


def get_ambient_resolved_config() -> ResolvedConfig | None:
    """Get the ambient resolved configuration if set by a scope.

    This function is used internally during configuration resolution.
    It only returns the ambient config if one is set by a scope, otherwise None.

    Returns:
        ResolvedConfig from the current scope, or None if no ambient config.

    Note:
        This is an internal function. The resolution fallback is handled
        by the caller (api.py) to avoid circular imports.
    """
    try:
        # If inside a config_scope, return the scoped config
        return _ambient_resolved_config.get()
    except LookupError:
        # No ambient config set - return None instead of resolving
        return None


@contextmanager
def config_scope(config: ResolvedConfig) -> Generator[None]:
    """Temporarily use a different resolved configuration.

    This context manager allows running specific operations with different
    configuration settings without affecting the global environment. It's
    async-safe and only affects configuration resolution at entry time.

    IMPORTANT: This only affects resolve_config() calls made within the scope.
    Once a FrozenConfig is attached to a command and enters the pipeline,
    handlers will not see ambient changes.

    Args:
        config: The ResolvedConfig to use within this scope

    Example:
        base_config = resolve_config()
        test_config = base_config.with_overrides(use_real_api=False)

        with config_scope(test_config):
            # Any resolve_config() calls here will use test_config
            executor = create_executor()  # Gets test_config
            await executor.execute(...)
    """
    token = _ambient_resolved_config.set(config)
    try:
        yield
    finally:
        _ambient_resolved_config.reset(token)


@contextmanager
def config_override(**overrides: Any) -> Generator[None]:
    """Convenience context manager for programmatic config overrides.

    This is a more convenient way to temporarily override specific configuration
    values without manually creating a ResolvedConfig.

    Args:
        **overrides: Configuration fields to override

    Example:
        with config_override(model="gemini-2.0-pro", use_real_api=False):
            # Code inside will use the overridden values
            config = resolve_config()  # Will have model="gemini-2.0-pro"
    """
    # Get current config and apply overrides
    base_config = get_ambient_resolved_config()
    if base_config is None:
        # Import here to avoid circular dependency at module level
        from .api import resolve_config

        base_config = resolve_config()
    scoped_config = base_config.with_overrides(**overrides)

    # Use the config_scope to apply the overridden config
    with config_scope(scoped_config):
        yield


# Updated resolve_config that considers ambient scope
def resolve_config_with_ambient(
    programmatic: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
    use_env_file: str | None = None,
    project_root: str | None = None,
) -> ResolvedConfig:
    """Resolve configuration considering ambient scope.

    This function checks if there's an ambient ResolvedConfig from config_scope(),
    and if so, applies any programmatic overrides to it. Otherwise, it performs
    normal resolution.

    Args:
        programmatic: Programmatic overrides to apply
        profile: Profile to use (ignored if ambient config exists)
        use_env_file: .env file path (ignored if ambient config exists)
        project_root: Project root path (ignored if ambient config exists)

    Returns:
        ResolvedConfig with proper precedence handling

    Note:
        This is used internally by the main resolve_config() function.
    """
    try:
        # If we have ambient config from a scope, use it as the base
        ambient_config = _ambient_resolved_config.get()

        if programmatic:
            # Apply programmatic overrides to the ambient config
            return ambient_config.with_overrides(**programmatic)
        return ambient_config

    except LookupError:
        # No ambient config, do normal resolution
        # Import here to avoid circular imports
        from .api import resolve_config as _resolve_config

        return _resolve_config(
            programmatic=programmatic,
            profile=profile,
            use_env_file=use_env_file,
            project_root=Path(project_root) if project_root else None,
        )
