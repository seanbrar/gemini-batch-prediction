"""
Main Gemini API client for content generation and batch processing
"""

import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union, Unpack

from google import genai
from google.genai import types

from gemini_batch.efficiency.metrics import extract_usage_metrics

from .client.cache_manager import CacheManager, CacheStrategy
from .client.configuration import (
    RateLimitConfig,
)
from .client.content_processor import ContentProcessor
from .client.error_handler import GenerationErrorHandler
from .client.prompt_builder import PromptBuilder
from .client.rate_limiter import RateLimiter
from .client.token_counter import TokenCounter
from .config import ConfigManager, GeminiConfig, get_config
from .constants import (
    DEFAULT_CACHE_TTL,
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    LARGE_TEXT_THRESHOLD,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)
from .exceptions import APIError
from .telemetry import TelemetryContext

log = logging.getLogger(__name__)


class GeminiClient:
    """
    A unified, zero-ceremony Gemini client that uses ambient configuration but
    allows for local overrides, providing a simple and flexible interface.
    """

    def __init__(self, **config_overrides: Unpack[GeminiConfig]):
        """
        Creates a client using the ambient configuration, with optional overrides.

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

        # 4. Initialize all components directly within this class
        self.tele = TelemetryContext()
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
        self.cache_manager: Optional[CacheManager] = None
        self.token_counter: Optional[TokenCounter] = None

        if self.config_manager.enable_caching:
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
            rate_config = self.config_manager.get_rate_limiter_config(
                self.config_manager.model
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

    def get_config_summary(self) -> Dict[str, Any]:
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
            }
        )

        # Add caching info if available
        if self.config_manager and self.cache_manager:
            summary["caching_thresholds"] = self.config_manager.get_caching_thresholds(
                self.config_manager.model
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
        if not self.config_manager.enable_caching or not self.token_counter:
            return False

        # Use direct content for token estimation
        estimate = self.token_counter.estimate_for_caching(
            self.config_manager.model, content, prefer_implicit
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
            self.config_manager.model, content, prefer_implicit
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
        return self.config_manager.get_caching_thresholds(self.config_manager.model)

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
        with self.tele.scope("client.generate_content"):
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
        with self.tele.scope("client.generate_batch"):
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
        """Main generation logic with retries and error handling"""
        with self.tele.scope(
            "client.execute_generation",
            attributes={
                "is_batch": is_batch,
                "num_prompts": len(prompts),
                "has_system_instruction": bool(system_instruction),
                "has_response_schema": bool(response_schema),
            },
        ) as ctx:
            # Process content into GenAI parts
            parts = self.content_processor.process(content)
            ctx.metric("num_parts", len(parts))

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
                        parts,
                        system_instruction,
                        return_usage,
                        response_schema,
                        **options,
                    )
                except Exception as e:
                    # Determine content type for better error context
                    content_type = self._get_content_type_description(content)
                    cache_enabled = self.cache_manager is not None
                    self.error_handler.handle_generation_error(
                        e, response_schema, content_type, cache_enabled
                    )

    def _should_attempt_caching(self, parts: List[types.Part]) -> bool:
        """Determine if caching should be attempted for the given parts"""
        if not self.config_manager.enable_caching or not self.token_counter:
            return False

        # Check for non-cacheable parts (e.g., FileData)
        if any(not isinstance(part, (str, types.Blob)) for part in parts):
            return False

        # Simple heuristic: check token count
        token_count = self.token_counter.count_tokens(self.config_manager.model, parts)
        caching_thresholds = self.config_manager.get_caching_thresholds(
            self.config_manager.model
        )
        min_tokens = caching_thresholds.get("explicit") or float("inf")
        return token_count >= min_tokens

    def _try_cached_generation(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Attempt to use the cache for generation"""
        if not self.cache_manager or not self.token_counter:
            return None

        # Analyze caching strategy
        token_count = self.token_counter.count_tokens(self.config_manager.model, parts)
        cache_info = self.config_manager.can_use_caching(
            self.config_manager.model, token_count
        )

        if not cache_info.get("supported"):
            return None

        strategy = cache_info["recommendation"]

        try:
            if strategy == CacheStrategy.IMPLICIT:
                return self._generate_with_implicit_cache(
                    parts, system_instruction, return_usage, response_schema, **options
                )
            elif strategy == CacheStrategy.EXPLICIT:
                return self._generate_with_explicit_cache(
                    parts,
                    strategy,
                    system_instruction,
                    return_usage,
                    response_schema,
                    **options,
                )
        except APIError as e:
            # If caching fails for a known reason, log and fallback
            log.warning("Cache generation failed, falling back to standard API: %s", e)
            return None  # Fallback to non-cached generation

        return None

    def _generate_with_implicit_cache(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content using implicit (automatic) caching"""
        optimized_parts = self._optimize_parts_for_implicit_cache(parts)

        def api_call():
            model = self.client.get_generative_model(
                model_name=self.config_manager.model,
                system_instruction=system_instruction,
            )
            return model.generate_content(optimized_parts, **options)

        response = self._api_call_with_retry(api_call)
        return self._process_response(response, return_usage, response_schema)

    def _generate_with_explicit_cache(
        self,
        parts: List[types.Part],
        cache_strategy: CacheStrategy,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content using an explicit cache"""
        if not self.cache_manager:
            raise APIError("Cache manager not available for explicit caching")

        # Create or get cache
        cached_content = self.cache_manager.create_or_get(
            parts, self.config_manager.model, strategy=cache_strategy
        )
        log.info("Using cache: %s", cached_content.name)

        # Separate prompt parts from content parts
        content_parts, prompt_parts = self._separate_cacheable_content(parts)

        # Use cache reference in generation
        response = self._generate_with_cache_reference(
            cached_content.name,
            prompt_parts,
            return_usage,
            response_schema,
            **options,
        )

        # Enhance with cache metrics
        if isinstance(response, dict) and "usage_metadata" in response:
            response["usage_metadata"] = self._enhance_usage_with_cache_metrics(
                response["usage_metadata"], cached_content
            )

        return response

    def _optimize_parts_for_implicit_cache(
        self, parts: List[types.Part]
    ) -> List[types.Part]:
        """Prepares parts for implicit caching by merging text parts"""
        if not parts:
            return []

        # Find the first text part to merge subsequent text parts into
        first_text_index = -1
        for i, part in enumerate(parts):
            if self._is_text_part(part):
                first_text_index = i
                break

        if first_text_index == -1:
            return parts  # No text parts to merge

        merged_text = ""
        new_parts = list(parts[:first_text_index])

        for part in parts[first_text_index:]:
            if self._is_text_part(part):
                merged_text += part.text
            else:
                if merged_text:
                    new_parts.append(merged_text)
                    merged_text = ""
                new_parts.append(part)

        if merged_text:
            new_parts.append(merged_text)

        return new_parts

    def _separate_cacheable_content(
        self, parts: List[types.Part]
    ) -> tuple[List[types.Part], List[types.Part]]:
        """Separate content parts from prompt parts for explicit caching"""
        # Heuristic: cache large text and all file parts. Keep small text as prompt.
        content_parts = []
        prompt_parts = []
        for part in parts:
            if self._is_large_text_part(part) or self._is_file_part(part):
                content_parts.append(part)
            else:
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
        """Generate content using a reference to a cache"""
        from google.generativeai.client import _cached_content_warning

        _cached_content_warning()

        model = self.client.get_generative_model(model_name=self.config_manager.model)
        cached_content = model.from_cached_content(cached_content_name=cache_name)

        def api_call():
            # Build config parameters
            config_params = {"response_mime_type": "text/plain"}
            if response_schema:
                config_params["response_mime_type"] = "application/json"
                options["generation_config"] = options.get("generation_config", {})
                options["generation_config"]["response_schema"] = response_schema

            return cached_content.generate_content(prompt_parts, **options)

        response = self._api_call_with_retry(api_call)
        return self._process_response(
            response, return_usage, response_schema, {"cache_name": cache_name}
        )

    def _generate_with_parts(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Low-level generation from processed parts"""
        model = self.client.get_generative_model(
            model_name=self.config_manager.model, system_instruction=system_instruction
        )

        def api_call():
            # Build config parameters properly
            config_params = {"response_mime_type": "text/plain"}
            gen_config = types.GenerationConfig()

            if response_schema:
                config_params["response_mime_type"] = "application/json"
                gen_config["response_schema"] = response_schema

            # Merge with any user-provided generation_config
            if "generation_config" in options:
                # Need to convert dict to GenerationConfig object if needed
                user_config = options["generation_config"]
                if isinstance(user_config, dict):
                    gen_config.update(user_config)
                elif isinstance(user_config, types.GenerationConfig):
                    gen_config = user_config  # Assume user-provided one is complete
                options["generation_config"] = gen_config

            return model.generate_content(parts, **options)

        response = self._api_call_with_retry(api_call)
        return self._process_response(response, return_usage, response_schema)

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
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Process GenAI response, extracting text or structured data"""
        with self.tele.scope(
            "client.process_response",
            attributes={
                "return_usage": return_usage,
                "has_schema": bool(response_schema),
            },
        ):
            # Extract usage metadata first
            usage_metadata = extract_usage_metrics(response)

            try:
                # Handle structured (JSON) response if schema is provided
                if response_schema:
                    if not response.parts:
                        raise APIError(
                            "API response is empty, expected structured data."
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
                    "Error processing API response: %s. Response: %s", e, response
                )
                raise APIError(f"Failed to process API response: {e}") from e
            finally:
                # Log usage if available
                if usage_metadata:
                    self.tele.metric(
                        "total_tokens", usage_metadata.get("total_tokens", 0)
                    )
                    self.tele.metric(
                        "billable_tokens", usage_metadata.get("billable_tokens", 0)
                    )

    def _enhance_usage_with_cache_metrics(
        self, usage_info: Dict[str, int], response
    ) -> Dict[str, int]:
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
        self, api_call_func: Callable, max_retries: int = MAX_RETRIES
    ):
        """Execute an API call with exponential backoff and retry"""
        with self.tele.scope("client.api_call_with_retry") as ctx:
            for attempt in range(max_retries + 1):
                try:
                    # Use rate limiter before making the call
                    with self.rate_limiter.request_context():
                        response = api_call_func()
                        # Track successful API call
                        ctx.metric("api_attempts", attempt + 1)
                        ctx.metric("api_success", 1)
                        return response
                except Exception as error:
                    if attempt == max_retries:
                        # Final attempt failed, log and re-raise
                        ctx.metric("api_attempts", attempt + 1)
                        ctx.metric("api_failures", 1)
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
                        ctx.metric("retryable_errors", 1)
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
                        ctx.metric("api_attempts", attempt + 1)
                        ctx.metric("non_retryable_errors", 1)
                        log.error(
                            "API call failed with non-retryable error.", exc_info=True
                        )
                        raise

    # Public cache management methods

    def cleanup_expired_caches(self) -> int:
        """Clean up all expired caches managed by the client."""
        with self.tele.scope("client.cleanup_expired_caches"):
            if self.cache_manager:
                return self.cache_manager.cleanup_expired_caches()
            return 0

    def get_cache_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cache metrics if caching is enabled."""
        with self.tele.scope("client.get_cache_metrics"):
            if self.cache_manager:
                return self.cache_manager.get_cache_metrics().to_dict()
            return None

    def list_active_caches(self) -> List[Dict[str, Any]]:
        """
        List all active (not expired) caches.
        NOTE: This can be an expensive operation.
        """
        with self.tele.scope("client.list_active_caches"):
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

    def _is_text_part(self, part) -> bool:
        """Check if a part is a simple text part."""
        # In the native library, text parts can be strings
        return isinstance(part, str) or (
            hasattr(part, "text") and not hasattr(part, "file_data")
        )

    def _is_file_part(self, part) -> bool:
        """Check if a part is a file/blob part."""
        return hasattr(part, "file_data") or isinstance(part, types.Blob)

    def _is_large_text_part(self, part, threshold=LARGE_TEXT_THRESHOLD) -> bool:
        """Check if a text part is considered large."""
        if not self._is_text_part(part):
            return False

        text_content = part if isinstance(part, str) else part.text
        # A simple character count heuristic
        # A more accurate method would be to use a tokenizer
        return len(text_content) > threshold

    # Deprecated properties for backwards compatibility
    # These should be removed in a future major version.

    @property
    def model_name(self) -> str:
        """Legacy property for model name"""
        return self.config_manager.model

    @property
    def api_key(self) -> str:
        """Legacy property for API key"""
        return self.config_manager.api_key

    @property
    def rate_limit_requests(self) -> int:
        """Legacy property for rate limit requests"""
        return self.rate_limiter.config.requests_per_minute

    @property
    def rate_limit_tokens(self) -> int:
        """Legacy property for rate limit tokens"""
        return self.rate_limiter.config.tokens_per_minute

    def __repr__(self):
        return f"<GeminiClient config={self.config_manager!r}>"
