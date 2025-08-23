"""Default prompt assembler implementation.

This module implements the pure function that assembles PromptBundle objects
from ResolvedCommand inputs and configuration. It handles precedence rules,
file reading, source-aware guidance, and advanced builder hooks while
maintaining invariants and purity.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemini_batch.core.exceptions import ConfigurationError
from gemini_batch.core.types import PromptBundle, ResolvedCommand

if TYPE_CHECKING:
    from gemini_batch.config import FrozenConfig


def assemble_prompts(
    command: ResolvedCommand,
) -> PromptBundle:
    """Assemble prompts from configuration into an immutable PromptBundle.

    This is a pure function that composes prompts according to the documented
    precedence rules and invariants. It supports:
    - Inline configuration (system, prefix, suffix)
    - File inputs (system_file, user_file)
    - Source-aware guidance (apply_if_sources, sources_block)
    - Advanced builder hooks (builder)

    Args:
        command: ResolvedCommand with initial prompts and configuration.

    Returns:
        PromptBundle with assembled prompts and provenance hints.

    Raises:
        ConfigurationError: If file reading fails, builder hook errors,
            or configuration is invalid.
    """
    config = command.initial.config
    initial_prompts = command.initial.prompts
    has_sources = bool(command.resolved_sources)

    # Extract prompts configuration from extra fields
    prompts_config = _extract_prompts_config(config)

    # Check for advanced builder hook first
    if builder_path := prompts_config.get("builder"):
        return _call_builder_hook(builder_path, command)

    # Default assembly logic
    return _assemble_default(
        initial_prompts,
        has_sources=has_sources,
        prompts_config=prompts_config,
    )


# --- Internal helpers ---


def _extract_prompts_config(config: FrozenConfig) -> dict[str, Any]:
    """Extract prompts configuration from config.extra with smart defaults."""
    prompts_config = {}

    # Look for prompts.* keys in config.extra
    for key, value in config.extra.items():
        if key.startswith("prompts."):
            # Remove "prompts." prefix to get the actual config key
            config_key = key[8:]  # len("prompts.") == 8
            prompts_config[config_key] = value

    return prompts_config


def _assemble_default(
    initial_prompts: tuple[str, ...],
    *,
    has_sources: bool,
    prompts_config: dict[str, Any],
) -> PromptBundle:
    """Assemble prompts using default configuration-driven logic."""
    hints: dict[str, Any] = {"has_sources": has_sources}

    # --- System instruction assembly ---
    system_instruction: str | None = None

    # Precedence: inline system > system_file
    if inline_system := prompts_config.get("system"):
        system_instruction = str(inline_system).strip()
        hints["system_from"] = "inline"
    elif system_file := prompts_config.get("system_file"):
        system_instruction = _read_prompt_file(system_file, prompts_config)
        hints["system_from"] = "system_file"
        hints["system_file"] = str(system_file)

    # Apply source-aware guidance to system instruction
    if (
        system_instruction is not None
        and prompts_config.get("apply_if_sources", False)
        and has_sources
    ):
        sources_block = prompts_config.get("sources_block", "")
        if sources_block:
            system_instruction = f"{system_instruction}\n\n{sources_block}".strip()
    elif (
        system_instruction is None
        and prompts_config.get("apply_if_sources", False)
        and has_sources
    ):
        # If no system instruction but we have sources_block, use it as system
        sources_block = prompts_config.get("sources_block", "")
        if sources_block:
            system_instruction = sources_block.strip()
            hints["system_from"] = "sources_block"

    # --- User prompts assembly ---
    user_prompts: tuple[str, ...]
    if initial_prompts:
        # Apply prefix/suffix to existing prompts
        prefix = prompts_config.get("prefix", "")
        suffix = prompts_config.get("suffix", "")

        user_prompts = tuple(
            f"{prefix}{prompt}{suffix}".strip() for prompt in initial_prompts
        )
        hints["user_from"] = "initial"
    else:
        # Use user_file only when no initial prompts
        user_file = prompts_config.get("user_file")
        if user_file:
            file_content = _read_prompt_file(user_file, prompts_config)
            user_prompts = (file_content,)
            hints["user_from"] = "user_file"
            hints["user_file"] = str(user_file)
        else:
            # No prompts provided - this is a configuration error
            raise ConfigurationError(
                "No prompts provided. Either pass prompts to InitialCommand or set prompts.user_file"
            )

    # Add telemetry hints
    if system_instruction:
        hints["system_len"] = len(system_instruction)
    hints["user_total_len"] = sum(len(p) for p in user_prompts)

    return PromptBundle(
        user=user_prompts,
        system=system_instruction,
        hints=hints,
    )


def _read_prompt_file(file_path: str | Path, prompts_config: dict[str, Any]) -> str:
    """Read a prompt file with just-in-time error handling."""
    path = Path(file_path)

    # Extract configuration with smart defaults
    encoding = prompts_config.get("encoding", "utf-8")
    strip_newlines = prompts_config.get("strip", True)
    max_bytes = prompts_config.get("max_bytes", 128_000)

    # Try to read the file - let failures happen naturally, but make them actionable
    try:
        content = path.read_text(encoding=encoding)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Prompt file '{path}' not found. Check the file path or create the file."
        ) from None
    except UnicodeDecodeError:
        raise ConfigurationError(
            f"Prompt file '{path}' encoding issue. "
            f"Try setting prompts.encoding to a different value (current: '{encoding}')."
        ) from None
    except PermissionError:
        raise ConfigurationError(
            f"Permission denied reading '{path}'. Check file permissions."
        ) from None
    except Exception as e:
        raise ConfigurationError(f"Failed to read prompt file '{path}': {e}") from e

    # Check size after reading to avoid race conditions
    if len(content.encode("utf-8")) > max_bytes:
        raise ConfigurationError(
            f"Prompt file '{path}' is too large ({len(content):,} chars). "
            f"Reduce file size or increase prompts.max_bytes (current: {max_bytes:,})."
        )

    # Process content
    if strip_newlines:
        content = content.rstrip("\n\r")

    # Return content even if empty - let caller decide if that's an error
    return content


def _call_builder_hook(builder_path: str, command: ResolvedCommand) -> PromptBundle:
    """Call an advanced builder hook function with minimal intervention."""
    try:
        # Parse dotted path: "pkg.mod:fn"
        if ":" not in builder_path:
            raise ConfigurationError(
                f"Invalid builder path '{builder_path}': use format 'module:function'"
            )

        module_path, function_name = builder_path.split(":", 1)
        module = importlib.import_module(module_path)
        builder_fn = getattr(module, function_name)

        # Call the builder - trust the type system and let failures be informative
        result = builder_fn(command)

        # Quick type check - if it's wrong, the error will be clear
        if not isinstance(result, PromptBundle):
            raise ConfigurationError(
                f"Builder '{builder_path}' returned {type(result).__name__}, expected PromptBundle"
            )

        return result

    except Exception as e:
        raise ConfigurationError(f"Builder '{builder_path}' failed: {e}") from e
