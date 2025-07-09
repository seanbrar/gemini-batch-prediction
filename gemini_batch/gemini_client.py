"""
Main Gemini API client for content generation and batch processing
"""

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

from google import genai
from google.genai import types

from gemini_batch.efficiency.metrics import extract_usage_metrics

from .client.cache_manager import CacheManager, CacheStrategy
from .client.configuration import ClientConfiguration, RateLimitConfig
from .client.content_processor import ContentProcessor
from .client.error_handler import GenerationErrorHandler
from .client.prompt_builder import PromptBuilder
from .client.rate_limiter import RateLimiter
from .client.token_counter import TokenCounter
from .config import ConfigManager
from .constants import (
    DEFAULT_CACHE_TTL,
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    LARGE_TEXT_THRESHOLD,
    MAX_RETRIES,
    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS,
    RETRY_BASE_DELAY,
)
from .exceptions import APIError

log = logging.getLogger(__name__)


class GeminiClient:
    """Main Gemini API client for content generation and batch processing"""

    def __init__(self, config: ClientConfiguration):
        """Initialize client with configuration"""
        self.config = config
        self.config.validate()
        log.debug(
            "GeminiClient initialized with model '%s'. Caching: %s.",
            self.config.model_name,
            self.config.enable_caching,
        )

        # Core components (always available)
        self.client = genai.Client(api_key=self.config.api_key)
        self.rate_limiter = RateLimiter(self._get_rate_limit_config())
        self.prompt_builder = PromptBuilder()
        self.error_handler = GenerationErrorHandler()
        self.content_processor = ContentProcessor()

        # Optional caching components with clear type hints
        self.cache_manager: Optional[CacheManager] = None
        self.config_manager: Optional[ConfigManager] = None
        self.token_counter: Optional[TokenCounter] = None

        # Initialize caching components if enabled
        if self.config.enable_caching:
            self.config_manager = ConfigManager(
                tier=self.config.tier,
                model=self.config.model_name,
                api_key=self.config.api_key,
            )
            self.token_counter = TokenCounter(self.client, self.config_manager)
            self.cache_manager = CacheManager(
                client=self.client,
                config_manager=self.config_manager,
                token_counter=self.token_counter,
                default_ttl_seconds=DEFAULT_CACHE_TTL,
            )

    def _get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limiting configuration for the current model"""
        try:
            # Create temporary ConfigManager for rate limit config
            temp_config = ConfigManager(
                api_key=self.config.api_key,
                model=self.config.model_name,
                tier=self.config.tier,
            )
            rate_config = temp_config.get_rate_limiter_config(self.config.model_name)
            return RateLimitConfig(
                requests_per_minute=rate_config["requests_per_minute"],
                tokens_per_minute=rate_config["tokens_per_minute"],
            )
        except Exception:
            return RateLimitConfig(
                requests_per_minute=FALLBACK_REQUESTS_PER_MINUTE,
                tokens_per_minute=FALLBACK_TOKENS_PER_MINUTE,
            )

    @classmethod
    def from_config_manager(
        cls, config_manager: ConfigManager, **overrides
    ) -> "GeminiClient":
        """Create client from ConfigManager with optional parameter overrides"""
        # Enhanced caching configuration
        enable_caching = overrides.get("enable_caching", False)

        # Auto-enable caching if using models that support it and not explicitly disabled
        if "enable_caching" not in overrides and config_manager.model:
            caching_support = config_manager.can_use_caching(config_manager.model, 4096)
            if caching_support.get("explicit") or caching_support.get("implicit"):
                enable_caching = True

        # Remove enable_caching from overrides to avoid duplicate parameter
        filtered_overrides = {
            k: v for k, v in overrides.items() if k != "enable_caching"
        }

        client_config = ClientConfiguration.from_config_manager(
            config_manager, enable_caching=enable_caching, **filtered_overrides
        )
        return cls(client_config)

    @classmethod
    def from_env(cls, **overrides) -> "GeminiClient":
        """Create client from environment variables with optional overrides"""
        # Handle direct API key parameter
        api_key = overrides.pop("api_key", None)

        if api_key:
            # Use parameters-based configuration when API key provided directly
            client_config = ClientConfiguration.from_parameters(
                api_key=api_key, **overrides
            )
            return cls(client_config)
        else:
            # Use environment-based configuration
            config_manager = ConfigManager.from_env()

            # Check for cache enabling environment variable
            import os

            env_caching = os.getenv("GEMINI_ENABLE_CACHING", "").lower() in (
                "true",
                "1",
                "yes",
            )
            if env_caching and "enable_caching" not in overrides:
                overrides["enable_caching"] = True

            return cls.from_config_manager(config_manager, **overrides)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration summary for debugging"""
        summary = self.config.get_summary()
        summary.update(
            {
                "rate_limiter_config": {
                    "requests_per_minute": self.rate_limiter.config.requests_per_minute,
                    "tokens_per_minute": self.rate_limiter.config.tokens_per_minute,
                    "window_seconds": self.rate_limiter.config.window_seconds,
                },
                "caching_enabled": self.config.enable_caching,
                "caching_available": self.config_manager is not None,
            }
        )

        # Add caching info if available
        if self.config_manager and self.cache_manager:
            summary["caching_thresholds"] = self.config_manager.get_caching_thresholds(
                self.config.model_name
            )

            # Add cache metrics
            cache_metrics = self.cache_manager.get_cache_metrics()
            summary["cache_metrics"] = {
                "active_caches": cache_metrics.active_caches,
                "total_caches_created": cache_metrics.total_caches,
                "cache_hits": cache_metrics.cache_hits,
                "cache_misses": cache_metrics.cache_misses,
                "total_cached_tokens": cache_metrics.total_cached_tokens,
            }
        else:
            summary["caching_thresholds"] = None
            summary["cache_metrics"] = None

        return summary

    def should_cache_content(self, content, prefer_implicit: bool = True) -> bool:
        """
        Simple caching decision - returns True if content should be cached.

        Returns False when caching is disabled - this is the expected behavior.
        """
        if not self.config.enable_caching or not self.token_counter:
            return False

        # Use direct content for token estimation
        estimate = self.token_counter.estimate_for_caching(
            self.config.model_name, content, prefer_implicit
        )

        return estimate["cacheable"]

    def analyze_caching_strategy(
        self, content, prefer_implicit: bool = True
    ) -> Dict[str, Any]:
        """
        Detailed caching analysis with strategy recommendations.

        Provides useful information even when caching is disabled.
        """
        if not self.token_counter:
            return {
                "tokens": 0,
                "cacheable": False,
                "strategy": "none",
                "details": {"reason": "Caching not enabled"},
                "caching_enabled": False,
            }

        estimate = self.token_counter.estimate_for_caching(
            self.config.model_name, content, prefer_implicit
        )

        return {
            "tokens": estimate["tokens"],
            "cacheable": estimate["cacheable"],
            "strategy": estimate["recommended_strategy"],
            "details": estimate,
            "caching_enabled": True,
        }

    def get_caching_thresholds(self) -> Optional[Dict[str, Optional[int]]]:
        """
        Get caching thresholds for the current model.

        Returns None when caching is disabled.
        """
        if not self.config_manager:
            return None
        return self.config_manager.get_caching_thresholds(self.config.model_name)

    def generate_content(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content from text, files, URLs, or mixed sources"""
        return self._execute_generation(
            content=content,
            prompts=[prompt] if prompt else [],
            system_instruction=system_instruction,
            return_usage=return_usage,
            response_schema=response_schema,
            is_batch=False,
            **options,
        )

    def generate_batch(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Process multiple questions about the same content"""
        if not questions:
            raise APIError("At least one question is required for batch processing")

        return self._execute_generation(
            content=content,
            prompts=questions,
            system_instruction=system_instruction,
            return_usage=return_usage,
            response_schema=response_schema,
            is_batch=True,
            **options,
        )

    def _execute_generation(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        prompts: List[str],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        is_batch: bool = False,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Core generation method for both single and batch requests"""
        # Get extracted content from content processor
        extracted_contents = self.content_processor.process_content(content)

        # Convert to API parts with caching-aware upload decisions
        parts = self.content_processor.create_api_parts(
            extracted_contents,
            cache_enabled=self.config.enable_caching,
            client=self.client,
        )

        # Add prompts using PromptBuilder
        if is_batch:
            batch_prompt = self.prompt_builder.create_batch_prompt(
                prompts, response_schema
            )
            parts.append(types.Part(text=batch_prompt))
        else:
            for prompt in prompts:
                if prompt:
                    clean_prompt = self.prompt_builder.create_single_prompt(prompt)
                    parts.append(types.Part(text=clean_prompt))

        # Execute API call with rate limiting
        with self.rate_limiter.request_context():
            try:
                # Try cache-aware generation if caching enabled
                if self.cache_manager and self._should_attempt_caching(parts):
                    cache_result = self._try_cached_generation(
                        parts,
                        system_instruction,
                        return_usage,
                        response_schema,
                        **options,
                    )
                    if cache_result is not None:
                        return cache_result

                # Fallback to existing generation flow
                return self._generate_with_parts(
                    parts, system_instruction, return_usage, response_schema, **options
                )
            except Exception as e:
                # Determine content type for better error context
                content_type = self._get_content_type_description(content)
                cache_enabled = self.cache_manager is not None
                self.error_handler.handle_generation_error(
                    e, response_schema, content_type, cache_enabled
                )

    def _should_attempt_caching(self, parts: List[types.Part]) -> bool:
        """Determine if caching should be attempted based on config and content"""
        if not self.config.enable_caching or not self.cache_manager:
            return False

        # Simplified check: if any part is a large text part, attempt caching.
        # This avoids re-implementing complex token counting here.
        # The cache_manager will perform the detailed analysis.
        for part in parts:
            if self._is_large_text_part(
                part, min_len=MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
            ):
                return True
        return False

    def _try_cached_generation(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Attempt cached generation with graceful fallback."""
        if not self.cache_manager:
            return None
        try:
            # Analyze caching strategy with correct parameter order
            cache_strategy = self.cache_manager.analyze_cache_strategy(
                model=self.config.model_name,
                content=parts,
                prefer_explicit=True,
            )
            log.debug(
                "Cache analysis result: strategy=%s, tokens=%d, reason='%s'",
                cache_strategy.strategy_type,
                cache_strategy.estimated_tokens,
                cache_strategy.reason,
            )

            if not cache_strategy.should_cache:
                log.debug("Content not suitable for caching, proceeding without cache.")
                return None  # Fall back to non-cached generation

            if cache_strategy.strategy_type == "explicit":
                log.info("Using explicit context caching for this request.")
                return self._generate_with_explicit_cache(
                    parts,
                    cache_strategy,
                    system_instruction,
                    return_usage,
                    response_schema,
                    **options,
                )

            # For implicit caching, structure content optimally and use regular flow
            elif cache_strategy.strategy_type == "implicit":
                return self._generate_with_implicit_cache(
                    parts, system_instruction, return_usage, response_schema, **options
                )

            # Unknown strategy, fall back
            return None

        except Exception as e:
            log.warning(
                "Cache generation attempt failed, falling back to standard generation. Reason: %s",
                e,
            )
            return None

    def _generate_with_implicit_cache(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate with implicit caching optimization.

        Structures content to maximize cache hit probability.
        """
        # Implicit caching optimization: put large/common content first
        optimized_parts = self._optimize_parts_for_implicit_cache(parts)

        return self._generate_with_parts(
            optimized_parts,
            system_instruction,
            return_usage,
            response_schema,
            **options,
        )

    def _generate_with_explicit_cache(
        self,
        parts: List[types.Part],
        cache_strategy: CacheStrategy,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate using explicit cache with lifecycle management"""

        # Separate cacheable content from prompt parts
        content_parts, prompt_parts = self._separate_cacheable_content(parts)

        if not content_parts:
            return None  # Nothing to cache, fall back

        # Get or create cache for content
        cache_result = self.cache_manager.get_or_create_cache(
            model=self.config.model_name,
            content_parts=content_parts,
            strategy=cache_strategy,
            system_instruction=system_instruction,
        )

        if not cache_result.success:
            return None  # Cache failed, fall back to regular generation

        # Generate with cached content
        return self._generate_with_cache_reference(
            cache_name=cache_result.cache_name,
            prompt_parts=prompt_parts,
            return_usage=return_usage,
            response_schema=response_schema,
            **options,
        )

    def _optimize_parts_for_implicit_cache(
        self, parts: List[types.Part]
    ) -> List[types.Part]:
        """Optimize part ordering for implicit cache hits"""
        # For implicit caching: large, stable content should come first
        file_parts = []
        text_parts = []
        prompt_parts = []

        for part in parts:
            if self._is_file_part(part):
                file_parts.append(part)  # Files first (usually largest)
            elif self._is_large_text_part(part):
                text_parts.append(part)  # Large text second
            else:
                prompt_parts.append(part)  # Prompts/questions last

        # Optimal order: files, large text, prompts
        return file_parts + text_parts + prompt_parts

    def _separate_cacheable_content(
        self, parts: List[types.Part]
    ) -> tuple[List[types.Part], List[types.Part]]:
        """Separate content parts (cacheable) from prompt parts"""
        content_parts = []
        prompt_parts = []

        for part in parts:
            # Files and large text are cacheable
            if self._is_file_part(part):
                content_parts.append(part)
            elif self._is_large_text_part(part):
                # Large text content is worth caching
                content_parts.append(part)
            else:
                # Short text (prompts/questions) not worth caching
                prompt_parts.append(part)

        return content_parts, prompt_parts

    def _generate_with_cache_reference(
        self,
        cache_name: str,
        prompt_parts: List[types.Part],
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content using explicit cache reference"""

        def api_call():
            # Build config parameters
            config_params = {"cached_content": cache_name}  # Cache reference

            if response_schema:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_schema

            config = types.GenerateContentConfig(**config_params)

            # Generate with cache reference and remaining prompts
            return self.client.models.generate_content(
                model=self.config.model_name,
                contents=types.Content(parts=prompt_parts),
                config=config,
            )

        response = self._api_call_with_retry(api_call)

        return self._process_response(
            response, return_usage, response_schema, {"model": self.config.model_name}
        )

    def _generate_with_parts(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Execute API call with prepared content parts"""

        def api_call():
            # Build config parameters properly
            config_params = {}

            if system_instruction:
                config_params["system_instruction"] = system_instruction

            if response_schema:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_schema

            # Create config only if we have parameters
            config = (
                types.GenerateContentConfig(**config_params) if config_params else None
            )

            return self.client.models.generate_content(
                model=self.config.model_name,
                contents=types.Content(parts=parts),
                config=config,
            )

        response = self._api_call_with_retry(api_call)

        return self._process_response(
            response, return_usage, response_schema, {"model": self.config.model_name}
        )

    def _get_content_type_description(self, content) -> str:
        """Get descriptive name for content type for error messages"""
        if isinstance(content, list):
            return f"mixed content ({len(content)} items)"
        elif isinstance(content, Path):
            return f"file ({content.suffix or 'no extension'})"
        elif isinstance(content, str):
            if content.startswith(("http://", "https://")):
                return "URL"
            elif len(content) < 100:
                return "text"
            else:
                return "long text"
        else:
            return "unknown content type"

    def _process_response(
        self,
        response,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Process API response with optional usage tracking and validation"""

        # Handle structured output with simple validation
        if response_schema:
            # Simple validation using Pydantic
            try:
                if hasattr(response, "parsed") and response.parsed:
                    parsed_data = response.parsed
                    success = True
                    errors = []
                else:
                    # Fallback to text parsing
                    response_text = getattr(response, "text", "")
                    import json

                    loaded_json = json.loads(response_text)
                    parsed_data = response_schema.model_validate(loaded_json)
                    success = True
                    errors = []
            except Exception as e:
                parsed_data = None
                success = False
                errors = [str(e)]

            if return_usage:
                # Extract and return with usage information
                usage_info = extract_usage_metrics(response)

                # Enhanced usage info includes cache metrics
                usage_info = self._enhance_usage_with_cache_metrics(
                    usage_info, response
                )

                result = {
                    "text": response.text,
                    "parsed": parsed_data,
                    "structured_success": success,
                    "structured_confidence": 0.9 if success else 0.3,
                    "validation_method": "simple_pydantic",
                    "validation_errors": errors,
                    "usage": usage_info,
                }

                # Add extra metadata if provided
                if extra_metadata:
                    result.update(extra_metadata)

                return result
            else:
                # Return best available data
                return parsed_data if success else response.text

        # Standard text response
        if return_usage:
            # Extract and return with usage information
            usage_info = extract_usage_metrics(response)

            # Enhanced usage info includes cache metrics
            usage_info = self._enhance_usage_with_cache_metrics(usage_info, response)

            result = {
                "text": response.text,
                "usage": usage_info,
            }

            # Add extra metadata if provided
            if extra_metadata:
                result.update(extra_metadata)

            return result
        else:
            # Just return the text
            return response.text

    def _enhance_usage_with_cache_metrics(
        self, usage_info: Dict[str, int], response
    ) -> Dict[str, int]:
        """Enhance usage info with cache-specific metrics"""
        # The enhanced usage_info already includes cached_content_token_count from extract_usage_metrics
        # Add cache efficiency metrics if available
        if hasattr(response, "usage_metadata") and hasattr(
            response.usage_metadata, "cached_content_token_count"
        ):
            cached_tokens = (
                getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
            )
            if cached_tokens > 0:
                # Calculate cache efficiency
                total_input = usage_info.get("prompt_tokens", 0)
                if total_input > 0:
                    cache_hit_ratio = cached_tokens / total_input
                    usage_info["cache_hit_ratio"] = round(cache_hit_ratio, 3)
                    usage_info["cache_enabled"] = True
                else:
                    usage_info["cache_enabled"] = False
            else:
                usage_info["cache_enabled"] = False

        return usage_info

    def _api_call_with_retry(
        self, api_call_func: Callable, max_retries: int = MAX_RETRIES
    ):
        """Execute API call with exponential backoff retry logic"""
        for attempt in range(max_retries + 1):
            try:
                # Use rate limiter before making the call
                with self.rate_limiter.request_context():
                    return api_call_func()
            except Exception as error:
                if attempt == max_retries:
                    # Final attempt failed, log and re-raise
                    log.error(
                        "API call failed after %d retries.", max_retries, exc_info=True
                    )
                    raise

                # Check if it's a retryable error
                error_str = str(error).lower()
                retryable_terms = [
                    "rate limit",
                    "429",
                    "quota",
                    "timeout",
                    "temporary",
                    "service unavailable",
                    "500",
                    "502",
                    "503",
                    "504",
                ]

                if any(term in error_str for term in retryable_terms):
                    # Calculate delay with exponential backoff
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    log.warning(
                        "API call failed with retryable error. Retrying in %.2fs (Attempt %d/%d)",
                        delay,
                        attempt + 2,
                        max_retries + 1,
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Non-retryable error, log and re-raise
                    log.error(
                        "API call failed with non-retryable error.", exc_info=True
                    )
                    raise

    # Public cache management methods

    def cleanup_expired_caches(self) -> int:
        """
        Clean up expired caches and return count of cleaned caches.

        Returns 0 when caching is disabled.
        """
        if not self.cache_manager:
            return 0
        return self.cache_manager.cleanup_expired_caches()

    def get_cache_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current cache usage metrics if caching is enabled.

        Returns None when caching is disabled.
        """
        if not self.cache_manager:
            return None

        metrics = self.cache_manager.get_cache_metrics()
        return {
            "total_caches_created": metrics.total_caches,
            "active_caches": metrics.active_caches,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "total_cached_tokens": metrics.total_cached_tokens,
            "cache_hit_rate": (
                metrics.cache_hits / max(metrics.cache_hits + metrics.cache_misses, 1)
            ),
            "average_creation_time": (
                metrics.cache_creation_time / max(metrics.total_caches, 1)
            ),
        }

    def list_active_caches(self) -> List[Dict[str, Any]]:
        """
        List active caches with summary information.

        Returns empty list when caching is disabled
        """
        if not self.cache_manager:
            return []

        cache_infos = self.cache_manager.list_active_caches()
        return [
            {
                "cache_name": info.cache_name,
                "model": info.model,
                "created_at": info.created_at.isoformat(),
                "ttl_seconds": info.ttl_seconds,
                "token_count": info.token_count,
                "usage_count": info.usage_count,
                "last_used": info.last_used.isoformat() if info.last_used else None,
            }
            for info in cache_infos
        ]

    # Helper methods for part type checking
    def _is_text_part(self, part) -> bool:
        """Check if part contains text content"""
        return (
            hasattr(part, "text")
            and part.text is not None
            and isinstance(part.text, str)
        )

    def _is_file_part(self, part) -> bool:
        """Check if part contains file data"""
        return hasattr(part, "file_data") and part.file_data is not None

    def _is_large_text_part(self, part, threshold=LARGE_TEXT_THRESHOLD) -> bool:
        """Check if part contains large text content"""
        if not self._is_text_part(part):
            return False

        # Handle non-string types safely
        try:
            text_length = len(part.text)
            return text_length > threshold
        except (TypeError, AttributeError):
            # Objects that don't support len() comparison
            return False

    @property
    def model_name(self) -> str:
        """Get the current model name"""
        return self.config.model_name

    @property
    def api_key(self) -> str:
        """Get the API key"""
        return self.config.api_key

    @property
    def rate_limit_requests(self) -> int:
        """Get the rate limit for requests per minute"""
        return self.rate_limiter.requests_per_minute

    @property
    def rate_limit_tokens(self) -> int:
        """Get the rate limit for tokens per minute"""
        return self.rate_limiter.tokens_per_minute
