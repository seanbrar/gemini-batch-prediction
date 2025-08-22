#!/usr/bin/env python3
"""Example demonstrating the new configuration system.

This example shows how to use the resolve-once, freeze-then-flow
configuration system with different resolution methods.
"""

import os

from gemini_batch.config import resolve_config
from gemini_batch.config.compatibility import ConfigCompatibilityShim
from gemini_batch.executor import create_executor


def main() -> None:
    """Demonstrate configuration usage patterns."""
    print("=== Configuration System Example ===\n")

    # 1. Basic configuration resolution
    print("1. Basic Configuration Resolution:")
    try:
        # Set a test API key for demonstration
        os.environ["GEMINI_API_KEY"] = "test-api-key-for-demo"

        resolved = resolve_config()
        frozen = resolved.to_frozen()

        print(f"   Model: {frozen.model}")
        print(f"   Tier: {frozen.tier.value}")
        print(f"   Caching: {frozen.enable_caching}")
        print(f"   API Key Set: {frozen.api_key is not None}")
        print(f"   Sources: {dict(resolved.origin)}")

    except Exception as e:
        print(f"   Error: {e}")

    print()

    # 2. Programmatic overrides
    print("2. Programmatic Configuration Override:")
    try:
        resolved = resolve_config(
            programmatic={"model": "gemini-1.5-flash", "enable_caching": False}
        )
        frozen = resolved.to_frozen()

        print(f"   Model: {frozen.model}")
        print(f"   Caching: {frozen.enable_caching}")
        print(f"   Model Source: {resolved.origin['model']}")
        print(f"   Caching Source: {resolved.origin['enable_caching']}")

    except Exception as e:
        print(f"   Error: {e}")

    print()

    # 3. Executor integration
    print("3. Executor Integration:")
    try:
        executor = create_executor()
        config_shim = ConfigCompatibilityShim(executor.config)

        print(f"   Executor Config Type: {type(executor.config).__name__}")
        print(f"   Model via Shim: {config_shim.model}")
        print(f"   API Key Set: {config_shim.api_key is not None}")

    except Exception as e:
        print(f"   Error: {e}")

    print()

    # 4. Configuration validation
    print("4. Configuration Validation:")
    try:
        from gemini_batch.config.introspection import get_config_info

        info = get_config_info()
        print(f"   Status: {info['status']}")
        print(f"   Warnings: {len(info['validation']['warnings'])}")
        print(f"   Errors: {len(info['validation']['errors'])}")

        if info["validation"]["warnings"]:
            for warning in info["validation"]["warnings"]:
                print(f"   Warning: {warning}")

    except Exception as e:
        print(f"   Error: {e}")

    print()
    print("=== Configuration Example Complete ===")

    # Clean up demo environment variable
    if (
        "GEMINI_API_KEY" in os.environ
        and os.environ["GEMINI_API_KEY"] == "test-api-key-for-demo"
    ):
        del os.environ["GEMINI_API_KEY"]


if __name__ == "__main__":
    main()
