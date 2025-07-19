"""Main Gemini API client for content generation and batch processing"""

from collections.abc import Callable
import logging
from pathlib import Path
import time
from typing import Any, Unpack

from google import genai
from google.genai import types

from gemini_batch.efficiency.metrics import extract_usage_metrics

from .client.cache_manager import CacheManager
from .client.configuration import (
    RateLimitConfig,
)
from .client.content_processor import ContentProcessor
from .client.error_handler import GenerationErrorHandler
from .client.models import CacheAction, CacheStrategy, PartsPayload
from .client.prompt_builder import PromptBuilder
from .client.rate_limiter import RateLimiter
from .client.token_counter import TokenCounter
from .config import ConfigManager, GeminiConfig, get_config
from .constants import (
    DEFAULT_CACHE_TTL,
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)
from .exceptions import APIError
from .telemetry import TelemetryContext, TelemetryContextProtocol

log = logging.getLogger(__name__)


class GeminiClient:
    """A unified, zero-ceremony Gemini client that uses ambient configuration but
    allows for local overrides, providing a simple and flexible interface.
    """

    def __init__(
        self,
        telemetry_context: TelemetryContextProtocol | None = None,
        **config_overrides: Unpack[GeminiConfig],
    ):
        """Creates a client using the ambient configuration, with optional overrides.

        Examples:
            client = GeminiClient()  # Uses ambient/env config
            client = GeminiClient(model="gemini-2.5-pro")  # Overrides just the model
        """
        # 1. Start with the ambient configuration as a base
        base_config = get_config()

        # 2. Merge overrides into a dictionary. Overrides take precedence.
        merged_config_dict = {
            "api_key": base_config.api_key,
            "model": base_config.model,
            "tier": base_config.tier,
            "enable_caching": base_config.enable_caching,
            **config_overrides,
        }

        # 3. Create a single, definitive ConfigManager for this instance
        self.config_manager = ConfigManager(**merged_config_dict)

        self.tele = telemetry_context or TelemetryContext()

        log.debug(
            "GeminiClient initialized with model '%s'. Caching: %s.",
            self.config_manager.model,
            self.config_manager.enable_caching,
        )

        # Core components
        self.client = genai.Client(api_key=self.config_manager.api_key)
        self.rate_limiter = RateLimiter(self._get_rate_limit_config())
        self.prompt_builder = PromptBuilder()
        self.error_handler = GenerationErrorHandler()
        self.content_processor = ContentProcessor()

        # Optional caching components
        self.cache_manager: CacheManager | None = None
        self.token_counter: TokenCounter | None = None

        if self.config_manager.enable_caching:
            self.token_counter = TokenCounter(self.client, self.config_manager)
            self.cache_manager = CacheManager(
                client=self.client,
                config_manager=self.config_manager,
                token_counter=self.token_counter,
                default_ttl_seconds=DEFAULT_CACHE_TTL,
                content_processor=self.content_processor,
            )

    def _get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limiting configuration for the current model"""
        try:
            rate_config = self.config_manager.get_rate_limiter_config(
                self.config_manager.model,
            )
            return RateLimitConfig(
                requests_per_minute=rate_config["requests_per_minute"],
                tokens_per_minute=rate_config["tokens_per_minute"],
            )
        except Exception:
            return RateLimitConfig(
                requests_per_minute=FALLBACK_REQUESTS_PER_MINUTE,
                tokens_per_minute=FALLBACK_TOKENS_PER_MINUTE,
            )

    def get_config_summary(self) -> dict[str, Any]:
        """Get current configuration summary for debugging"""
        summary = self.config_manager.get_config_summary()
        summary.update(
            {
                "rate_limiter_config": {
                    "requests_per_minute": self.rate_limiter.config.requests_per_minute,
                    "tokens_per_minute": self.rate_limiter.config.tokens_per_minute,
                    "window_seconds": self.rate_limiter.config.window_seconds,
                },
                "caching_enabled": self.config_manager.enable_caching,
                "caching_available": self.config_manager is not None,
            },
        )

        # Add caching info if available
        if self.config_manager and self.cache_manager:
            summary["caching_thresholds"] = self.config_manager.get_caching_thresholds(
                self.config_manager.model,
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
        """Simple caching decision - returns True if content should be cached.

        Returns False when caching is disabled - this is the expected behavior.
        """
        if not self.config_manager.enable_caching or not self.token_counter:
            return False

        # Use direct content for token estimation
        estimate = self.token_counter.estimate_for_caching(
            self.config_manager.model,
            content,
            prefer_implicit,
        )

        return estimate["cacheable"]

    def analyze_caching_strategy(
        self,
        content,
        prefer_implicit: bool = True,
    ) -> dict[str, Any]:
        """Detailed caching analysis with strategy recommendations.

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
            self.config_manager.model,
            content,
            prefer_implicit,
        )

        return {
            "tokens": estimate["tokens"],
            "cacheable": estimate["cacheable"],
            "strategy": estimate["recommended_strategy"],
            "details": estimate,
            "caching_enabled": True,
        }

    def get_caching_thresholds(self) -> dict[str, int | None] | None:
        """Get caching thresholds for the current model.

        Returns None when caching is disabled.
        """
        if not self.config_manager:
            return None
        return self.config_manager.get_caching_thresholds(self.config_manager.model)

    def generate_content(
        self,
        content: str | Path | list[str | Path],
        prompt: str | None = None,
        system_instruction: str | None = None,
        return_usage: bool = False,
        response_schema: Any | None = None,
        **options,
    ) -> str | dict[str, Any]:
        """Generate content from text, files, URLs, or mixed sources"""
        with self.tele("client.generate_content"):
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
        content: str | Path | list[str | Path],
        questions: list[str],
        system_instruction: str | None = None,
        return_usage: bool = False,
        response_schema: Any | None = None,
        **options,
    ) -> str | dict[str, Any]:
        """Process multiple questions about the same content"""
        with self.tele("client.generate_batch"):
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
        content: str | Path | list[str | Path],
        prompts: list[str],
        system_instruction: str | None = None,
        return_usage: bool = False,
        response_schema: Any | None = None,
        is_batch: bool = False,
        **options,
    ) -> str | dict[str, Any]:
        """Main generation logic with retries and error handling"""
        with self.tele(
            "client.execute_generation",
            attributes={
                "is_batch": is_batch,
                "num_prompts": len(prompts),
                "has_system_instruction": bool(system_instruction),
                "has_response_schema": bool(response_schema),
            },
        ) as ctx:
            # Handle prompts by combining them with content
            if prompts:
                if is_batch:
                    # For batch processing, create a combined prompt
                    combined_prompt = self.prompt_builder.create_batch_prompt(
                        prompts,
                        response_schema,
                    )
                    # Add the combined prompt as text content
                    if isinstance(content, str):
                        content = f"{content}\n\n{combined_prompt}"
                    elif isinstance(content, list):
                        # Add prompt to the end of the content list
                        content = content + [combined_prompt]
                    else:
                        # For file content, add prompt as additional text
                        content = [content, combined_prompt]
                else:
                    # For single prompt, add it to content
                    single_prompt = self.prompt_builder.create_single_prompt(prompts[0])
                    if isinstance(content, str):
                        content = f"{content}\n\n{single_prompt}"
                    elif isinstance(content, list):
                        content = content + [single_prompt]
                    else:
                        content = [content, single_prompt]

            # ContentProcessor prepares the initial parts list.
            parts = self.content_processor.process(content, self.client)
            self.tele.metric("num_parts", len(parts))

            try:
                # 1. Get the plan from the Planner (CacheManager)
                if self.cache_manager:
                    action = self.cache_manager.plan_generation(parts)
                else:
                    # Fallback when caching is disabled
                    action = CacheAction(
                        CacheStrategy.GENERATE_RAW,
                        PartsPayload(parts=parts),
                    )

                # 2. Execute the plan based on the strategy
                if action.strategy == CacheStrategy.GENERATE_RAW:
                    return self._generate_with_parts(
                        action.payload.parts,  # Raw parts
                        system_instruction=system_instruction,
                        return_usage=return_usage,
                        response_schema=response_schema,
                        **options,
                    )

                if action.strategy == CacheStrategy.GENERATE_WITH_OPTIMIZED_PARTS:
                    return self._generate_with_parts(
                        action.payload.parts,  # Optimized parts
                        system_instruction=system_instruction,
                        return_usage=return_usage,
                        response_schema=response_schema,
                        **options,
                    )

                if action.strategy == CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE:
                    return self._generate_with_cache_reference(
                        cache_name=action.payload.cache_name,
                        prompt_parts=action.payload.parts,
                        return_usage=return_usage,
                        response_schema=response_schema,
                        **options,
                    )

                # Fallback for unknown strategies
                return self._generate_with_parts(
                    parts,
                    system_instruction=system_instruction,
                    return_usage=return_usage,
                    response_schema=response_schema,
                    **options,
                )

            except Exception as e:
                # The client is responsible for top-level error handling.
                content_type = self._get_content_type_description(content)
                cache_enabled = self.cache_manager is not None
                self.error_handler.handle_generation_error(
                    e,
                    response_schema,
                    content_type,
                    cache_enabled,
                )

    def _generate_with_parts(
        self,
        parts: list[types.Part],
        system_instruction: str | None = None,
        return_usage: bool = False,
        response_schema: Any | None = None,
        **options,
    ) -> str | dict[str, Any]:
        """Low-level generation from processed parts"""

        def api_call():
            # Build configuration properly
            config = types.GenerateContentConfig()

            if system_instruction:
                config.system_instruction = system_instruction

            if response_schema:
                config.response_mime_type = "application/json"
                config.response_schema = response_schema
            else:
                config.response_mime_type = "text/plain"

            # Merge with any user-provided generation_config
            if "generation_config" in options:
                user_config = options.pop("generation_config")  # Remove from options
                if isinstance(user_config, dict):
                    # Merge user config into our config
                    for key, value in user_config.items():
                        setattr(config, key, value)
                elif isinstance(user_config, types.GenerationConfig):
                    config = user_config  # Use user's config entirely

            # Use the new API format
            return self.client.models.generate_content(
                model=self.config_manager.model,
                contents=parts,
                config=config,
                **options,
            )

        response = self._api_call_with_retry(api_call)
        return self._process_response(response, return_usage, response_schema)

    def _generate_with_cache_reference(
        self,
        cache_name: str,
        prompt_parts: list[types.Part],
        return_usage: bool = False,
        response_schema: Any | None = None,
        **options,
    ) -> str | dict[str, Any]:
        """Generate content using explicit cache reference"""

        def api_call():
            # Build configuration properly
            config = types.GenerateContentConfig()
            config.cached_content = cache_name

            if response_schema:
                config.response_mime_type = "application/json"
                config.response_schema = response_schema
            else:
                config.response_mime_type = "text/plain"

            # Merge with any user-provided generation_config
            if "generation_config" in options:
                user_config = options.pop("generation_config")  # Remove from options
                if isinstance(user_config, dict):
                    # Merge user config into our config
                    for key, value in user_config.items():
                        setattr(config, key, value)
                elif isinstance(user_config, types.GenerationConfig):
                    config = user_config  # Use user's config entirely

            # Use the new API format
            return self.client.models.generate_content(
                model=self.config_manager.model,
                contents=prompt_parts,
                config=config,
                **options,
            )

        response = self._api_call_with_retry(api_call)
        return self._process_response(
            response,
            return_usage,
            response_schema,
            {"model": self.config_manager.model},
        )

    def _get_content_type_description(self, content) -> str:
        """Get a user-friendly description of the content type"""
        if isinstance(content, str):
            return "text"
        if isinstance(content, Path):
            return f"file ({content.suffix})"
        if isinstance(content, list):
            types_in_list = {type(item).__name__ for item in content}
            if len(types_in_list) == 1:
                return f"list of {types_in_list.pop()}"
            return "mixed content list"
        return "unknown"

    def _process_response(
        self,
        response,
        return_usage: bool = False,
        response_schema: Any | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Process GenAI response, extracting text or structured data"""
        log.debug("Entering telemetry scope: client.process_response")
        with self.tele(
            "client.process_response",
            attributes={
                "return_usage": return_usage,
                "has_schema": bool(response_schema),
            },
        ):
            # Handle case where response is already a processed dictionary
            if (
                isinstance(response, dict)
                and "text" in response
                and "usage_metadata" in response
            ):
                if return_usage:
                    return response
                return response["text"]

            # Extract usage metadata first
            usage_metadata = extract_usage_metrics(response)

            try:
                # Handle structured (JSON) response if schema is provided
                if response_schema:
                    if not response.parts:
                        raise APIError(
                            "API response is empty, expected structured data.",
                        )
                    # Assuming the first part contains the JSON data
                    data = response.parts[0].text
                    # The SDK should have already parsed this, but as a fallback:
                    if isinstance(data, str):
                        try:
                            import json

                            data = json.loads(data)
                        except json.JSONDecodeError as e:
                            raise APIError(f"Failed to parse JSON response: {e}") from e

                    if return_usage:
                        return {"data": data, "usage_metadata": usage_metadata}
                    return data

                # Handle standard text response
                text_response = response.text
                if return_usage:
                    return {"text": text_response, "usage_metadata": usage_metadata}
                return text_response

            except (ValueError, IndexError, AttributeError) as e:
                # Broad catch for unexpected response structures
                log.error(
                    "Error processing API response: %s. Response: %s",
                    e,
                    response,
                )
                raise APIError(f"Failed to process API response: {e}") from e
            finally:
                # Log usage if available
                if usage_metadata:
                    total_tokens_val = usage_metadata.get("total_tokens", 0)
                    billable_tokens_val = usage_metadata.get("billable_tokens", 0)

                    log.debug(
                        "Emitting telemetry metric: total_tokens = %d",
                        total_tokens_val,
                    )
                    self.tele.metric("total_tokens", total_tokens_val)

                    log.debug(
                        "Emitting telemetry metric: billable_tokens = %d",
                        billable_tokens_val,
                    )
                    self.tele.metric("billable_tokens", billable_tokens_val)

    def _enhance_usage_with_cache_metrics(
        self,
        usage_info: dict[str, int],
        response,
    ) -> dict[str, int]:
        """Enhance usage metadata with cache-specific metrics"""
        if hasattr(response, "usage_metadata"):
            cache_usage = response.usage_metadata
            usage_info["cached_content_token_count"] = (
                cache_usage.cached_content_token_count
            )
            # Update total tokens with cached content tokens
            usage_info["total_tokens"] = cache_usage.total_token_count

            # Add cache hit/miss info
            if self.cache_manager:
                metrics = self.cache_manager.get_cache_metrics()
                usage_info["cache_hits"] = metrics.cache_hits
                usage_info["cache_misses"] = metrics.cache_misses

        return usage_info

    def _api_call_with_retry(
        self,
        api_call_func: Callable,
        max_retries: int = MAX_RETRIES,
    ):
        """Execute an API call with exponential backoff and retry"""
        log.debug("Preparing to enter telemetry scope: client.api_call_with_retry")
        with self.tele("client.api_call_with_retry") as ctx:
            for attempt in range(max_retries + 1):
                try:
                    # Use rate limiter before making the call
                    with self.rate_limiter.request_context():
                        response = api_call_func()
                        # Track successful API call
                        log.debug("Emitting telemetry metric: api_success = 1")
                        self.tele.metric("api_success", 1)
                        return response
                except Exception as error:
                    if attempt == max_retries:
                        # Final attempt failed, log and re-raise
                        log.debug("Emitting telemetry metric: api_failures = 1")
                        self.tele.metric("api_failures", 1)
                        log.error(
                            "API call failed after %d retries.",
                            max_retries,
                            exc_info=True,
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
                        log.debug("Emitting telemetry metric: retryable_errors = 1")
                        self.tele.metric("retryable_errors", 1)
                        log.warning(
                            "API call failed with retryable error. Retrying in %.2fs (Attempt %d/%d)",
                            delay,
                            attempt + 2,
                            max_retries + 1,
                        )
                        time.sleep(delay)
                        continue
                    # Non-retryable error, log and re-raise
                    log.debug("Emitting telemetry metric: non_retryable_errors = 1")
                    self.tele.metric("non_retryable_errors", 1)
                    log.error(
                        "API call failed with non-retryable error.",
                        exc_info=True,
                    )
                    raise

    # Public cache management methods

    def cleanup_expired_caches(self) -> int:
        """Clean up all expired caches managed by the client."""
        with self.tele("client.cleanup_expired_caches"):
            if self.cache_manager:
                return self.cache_manager.cleanup_expired_caches()
            return 0

    def get_cache_metrics(self) -> dict[str, Any] | None:
        """Get cache metrics if caching is enabled."""
        with self.tele("client.get_cache_metrics"):
            if self.cache_manager:
                return self.cache_manager.get_cache_metrics().to_dict()
            return None

    def list_active_caches(self) -> list[dict[str, Any]]:
        """List all active (not expired) caches.
        NOTE: This can be an expensive operation.
        """
        with self.tele("client.list_active_caches"):
            if self.cache_manager:
                return [
                    {
                        "name": c.name,
                        "display_name": c.display_name,
                        "create_time": c.create_time,
                        "expire_time": c.expire_time,
                        "size_bytes": c.size_bytes,
                    }
                    for c in self.cache_manager.list_all()
                ]
            return []

    def __repr__(self):
        return f"<GeminiClient config={self.config_manager!r}>"
