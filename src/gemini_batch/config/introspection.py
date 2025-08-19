"""Configuration introspection utilities for debugging and validation.

This module provides utilities to inspect the effective configuration,
validate settings, and debug configuration resolution issues.
"""

import json
import sys
from typing import Any

from .api import resolve_config
from .types import ResolvedConfig

# ruff: noqa: T201


def print_config_debug(
    *,
    profile: str | None = None,
    show_sources: bool = True,
    show_validation: bool = True,
    programmatic_overrides: dict[str, Any] | None = None,
) -> None:
    """Print the effective configuration with sources and validation info.

    Args:
        profile: Optional profile name to use for resolution
        show_sources: Whether to show where each config value came from
        show_validation: Whether to run and show validation results
        programmatic_overrides: Optional programmatic config overrides
    """
    try:
        # Resolve configuration
        resolved = resolve_config(programmatic=programmatic_overrides, profile=profile)

        print("=== Effective Configuration ===")
        _print_config_values(resolved)

        if show_sources:
            print("\n=== Configuration Sources ===")
            _print_config_sources(resolved)

        if show_validation:
            print("\n=== Validation Results ===")
            _print_validation_results(resolved)

    except Exception as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)


def check_config_validation(
    *,
    profile: str | None = None,
    programmatic_overrides: dict[str, Any] | None = None,
) -> bool:
    """Check if configuration is valid and return success status.

    Args:
        profile: Optional profile name to use for resolution
        programmatic_overrides: Optional programmatic config overrides

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        resolve_config(programmatic=programmatic_overrides, profile=profile)
        return True
    except Exception:
        return False


def get_config_info() -> dict[str, Any]:
    """Get structured configuration information for programmatic use.

    Returns:
        Dictionary containing configuration details, sources, and validation status
    """
    try:
        resolved = resolve_config()

        return {
            "status": "valid",
            "config": {
                "model": resolved.model,
                "tier": resolved.tier.value,
                "enable_caching": resolved.enable_caching,
                "use_real_api": resolved.use_real_api,
                "ttl_seconds": resolved.ttl_seconds,
                "has_api_key": resolved.api_key is not None,
            },
            "sources": dict(resolved.origin),
            "validation": {
                "errors": [],
                "warnings": _get_config_warnings(resolved),
            },
        }
    except Exception as e:
        return {
            "status": "invalid",
            "error": str(e),
            "config": None,
            "sources": {},
            "validation": {
                "errors": [str(e)],
                "warnings": [],
            },
        }


def _print_config_values(resolved: ResolvedConfig) -> None:
    """Print the configuration values in a readable format."""
    frozen = resolved.to_frozen()

    print(f"  model: {frozen.model}")
    print(f"  tier: {frozen.tier.value}")
    print(f"  enable_caching: {frozen.enable_caching}")
    print(f"  use_real_api: {frozen.use_real_api}")
    print(f"  ttl_seconds: {frozen.ttl_seconds}")
    print(f"  api_key: {'[SET]' if frozen.api_key else '[NOT SET]'}")


def _print_config_sources(resolved: ResolvedConfig) -> None:
    """Print where each configuration value came from."""
    for field, source in resolved.origin.items():
        print(f"  {field}: {source}")


def _print_validation_results(resolved: ResolvedConfig) -> None:
    """Print validation results and any warnings."""
    print("✅ Configuration is valid")

    warnings = _get_config_warnings(resolved)
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("ℹ️  No warnings")  # noqa: RUF001


def _get_config_warnings(resolved: ResolvedConfig) -> list[str]:
    """Get configuration warnings (non-fatal issues)."""
    warnings = []

    # Check for common issues
    if not resolved.api_key:
        warnings.append("No API key configured - only mock responses will work")

    if resolved.tier.value == "free" and resolved.use_real_api:
        warnings.append(
            "Using 'free' tier with real API - rate limits will be very restrictive"
        )

    if resolved.ttl_seconds < 300:  # Less than 5 minutes
        warnings.append("TTL is very short - cached content will expire quickly")

    if resolved.ttl_seconds > 86400:  # More than 1 day
        warnings.append("TTL is very long - cached content may become stale")

    return warnings


def main() -> None:
    """CLI entry point for configuration introspection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect gemini-batch configuration",
        prog="python -m gemini_batch.config.introspection",
    )
    parser.add_argument("--profile", help="Configuration profile to use")
    parser.add_argument(
        "--no-sources", action="store_true", help="Don't show configuration sources"
    )
    parser.add_argument(
        "--no-validation", action="store_true", help="Don't show validation results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just check if configuration is valid (exit code 0=valid, 1=invalid)",
    )

    args = parser.parse_args()

    if args.check:
        # Simple validation check
        is_valid = check_config_validation(profile=args.profile)
        sys.exit(0 if is_valid else 1)

    if args.json:
        # JSON output for programmatic use
        info = get_config_info()
        print(json.dumps(info, indent=2))
    else:
        # Human-readable output
        print_config_debug(
            profile=args.profile,
            show_sources=not args.no_sources,
            show_validation=not args.no_validation,
        )


if __name__ == "__main__":
    main()
