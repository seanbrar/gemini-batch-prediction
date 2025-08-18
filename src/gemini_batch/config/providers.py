"""Provider-specific configuration registry.

This module implements the provider config registry system that allows
vendor-specific configuration views to be constructed from the core
FrozenConfig at the adapter boundary.
"""

from typing import Any, Protocol

from .types import FrozenConfig


class ProviderConfigBuilder(Protocol):
    """Protocol for provider-specific configuration builders.

    Provider adapters can implement this protocol to create vendor-specific
    configuration objects from the core FrozenConfig.
    """

    def build(self, base_config: FrozenConfig) -> Any:
        """Build a provider-specific configuration from the base config.

        Args:
            base_config: The core frozen configuration

        Returns:
            Provider-specific configuration object

        Example:
            class GeminiConfigBuilder:
                def build(self, base_config: FrozenConfig) -> GeminiClientConfig:
                    return GeminiClientConfig(
                        api_key=base_config.api_key,
                        model=base_config.model,
                        # Add provider-specific fields
                        request_timeout=30,
                        retry_count=3
                    )
        """


class ProviderConfigRegistry:
    """Registry for provider-specific configuration builders.

    This registry allows vendor adapters to register builders that can
    create provider-specific configuration objects from the core FrozenConfig.
    """

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._builders: dict[str, ProviderConfigBuilder] = {}

    def register(self, provider: str, builder: ProviderConfigBuilder) -> None:
        """Register a configuration builder for a provider.

        Args:
            provider: The provider name (e.g., "gemini", "openai", "anthropic")
            builder: The builder that implements ProviderConfigBuilder protocol

        Example:
            registry = ProviderConfigRegistry()
            registry.register("gemini", GeminiConfigBuilder())

            # Later, in the adapter
            gemini_config = registry.build("gemini", frozen_config)
        """
        self._builders[provider] = builder

    def unregister(self, provider: str) -> bool:
        """Unregister a provider's configuration builder.

        Args:
            provider: The provider name to unregister

        Returns:
            True if the provider was registered and removed, False otherwise
        """
        if provider in self._builders:
            del self._builders[provider]
            return True
        return False

    def build(self, provider: str, base_config: FrozenConfig) -> Any:
        """Build a provider-specific configuration.

        Args:
            provider: The provider name
            base_config: The core frozen configuration

        Returns:
            Provider-specific configuration object

        Raises:
            ValueError: If the provider is not registered

        Example:
            gemini_config = registry.build("gemini", frozen_config)
        """
        if provider not in self._builders:
            available = list(self._builders.keys())
            raise ValueError(
                f"Provider '{provider}' not registered. "
                f"Available providers: {available}"
            )

        builder = self._builders[provider]
        return builder.build(base_config)

    def is_registered(self, provider: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider: The provider name to check

        Returns:
            True if the provider is registered, False otherwise
        """
        return provider in self._builders

    def list_providers(self) -> list[str]:
        """List all registered providers.

        Returns:
            List of registered provider names
        """
        return list(self._builders.keys())

    def clear(self) -> None:
        """Clear all registered providers.

        This is primarily useful for testing.
        """
        self._builders.clear()


# Global registry instance
_global_registry = ProviderConfigRegistry()


def register_provider(provider: str, builder: ProviderConfigBuilder) -> None:
    """Register a provider configuration builder globally.

    This is a convenience function for registering with the global registry.

    Args:
        provider: The provider name
        builder: The configuration builder

    Example:
        from gemini_batch.config import register_provider

        register_provider("gemini", GeminiConfigBuilder())
    """
    _global_registry.register(provider, builder)


def build_provider_config(provider: str, base_config: FrozenConfig) -> Any:
    """Build a provider-specific configuration using the global registry.

    This is a convenience function for building with the global registry.

    Args:
        provider: The provider name
        base_config: The core frozen configuration

    Returns:
        Provider-specific configuration object

    Raises:
        ValueError: If the provider is not registered

    Example:
        # In an adapter
        gemini_config = build_provider_config("gemini", command.config)
    """
    return _global_registry.build(provider, base_config)


def get_provider_registry() -> ProviderConfigRegistry:
    """Get the global provider configuration registry.

    This allows access to the global registry for advanced use cases.

    Returns:
        The global ProviderConfigRegistry instance

    Example:
        registry = get_provider_registry()
        if registry.is_registered("gemini"):
            gemini_config = registry.build("gemini", frozen_config)
    """
    return _global_registry


def list_registered_providers() -> list[str]:
    """List all globally registered providers.

    Returns:
        List of registered provider names

    Example:
        providers = list_registered_providers()
        print(f"Available providers: {providers}")
    """
    return _global_registry.list_providers()


# Example provider config builder for Gemini
class GeminiProviderConfig:
    """Example provider-specific configuration for Gemini.

    This demonstrates how a provider might extend the core configuration
    with vendor-specific settings.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        tier: str,
        enable_caching: bool,
        use_real_api: bool,
        ttl_seconds: int,
        # Gemini-specific extensions
        request_timeout: int = 60,
        retry_count: int = 3,
        safety_settings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Gemini-specific configuration.

        Args:
            api_key: API key from base config
            model: Model from base config
            tier: Tier from base config
            enable_caching: Caching setting from base config
            use_real_api: Real API setting from base config
            ttl_seconds: TTL from base config
            request_timeout: Gemini-specific request timeout in seconds
            retry_count: Number of retries for failed requests
            safety_settings: Gemini safety settings configuration
        """
        # Core fields from FrozenConfig
        self.api_key = api_key
        self.model = model
        self.tier = tier
        self.enable_caching = enable_caching
        self.use_real_api = use_real_api
        self.ttl_seconds = ttl_seconds

        # Provider-specific extensions
        self.request_timeout = request_timeout
        self.retry_count = retry_count
        self.safety_settings = safety_settings or {}


class GeminiConfigBuilder:
    """Builder for Gemini-specific configuration.

    This demonstrates how to implement the ProviderConfigBuilder protocol.
    """

    def build(self, base_config: FrozenConfig) -> GeminiProviderConfig:
        """Build Gemini-specific configuration from base config.

        Args:
            base_config: The core frozen configuration

        Returns:
            GeminiProviderConfig with provider-specific extensions
        """
        return GeminiProviderConfig(
            api_key=base_config.api_key,
            model=base_config.model,
            tier=base_config.tier.value,  # Convert enum to string
            enable_caching=base_config.enable_caching,
            use_real_api=base_config.use_real_api,
            ttl_seconds=base_config.ttl_seconds,
            # Provider-specific defaults
            request_timeout=60,
            retry_count=3,
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            },
        )
