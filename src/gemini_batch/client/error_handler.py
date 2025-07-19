"""Error handling for Gemini API generation requests"""

from typing import Any

from ..exceptions import APIError


class GenerationErrorHandler:
    """Handles and enriches errors from Gemini API generation requests"""

    def handle_generation_error(
        self,
        error: Exception,
        response_schema: Any | None = None,
        content_type: str = "unknown",
        cache_enabled: bool = False,
    ) -> None:
        """Transform API errors into informative error messages"""
        error_str = str(error).lower()

        # Cache-specific error handling
        if cache_enabled and any(
            cache_term in error_str for cache_term in ["cache", "cached_content"]
        ):
            if "not found" in error_str or "expired" in error_str:
                raise APIError(
                    f"Cache expired or not found. Falling back to non-cached generation. "
                    f"Original error: {error}",
                ) from error
            if "ttl" in error_str:
                raise APIError(
                    f"Cache TTL configuration error. Check cache duration settings. "
                    f"Original error: {error}",
                ) from error
            if "cached_content" in error_str:
                raise APIError(
                    f"Cache content validation failed. Content may be too large or malformed. "
                    f"Original error: {error}",
                ) from error

        # Structured output specific errors
        if response_schema and ("json" in error_str or "schema" in error_str):
            cache_context = " (with caching enabled)" if cache_enabled else ""
            raise APIError(
                f"Structured output generation failed for {content_type} content{cache_context}. "
                f"Check your schema definition. Original error: {error}",
            ) from error

        # Content-specific errors with cache context
        cache_context = (
            " Note: Caching was enabled for this request." if cache_enabled else ""
        )

        if "quota" in error_str and "youtube" in error_str:
            raise APIError(
                f"YouTube quota exceeded. Free tier allows 8 hours/day.{cache_context} "
                f"Original error: {error}",
            ) from error
        if "private" in error_str or "unlisted" in error_str:
            raise APIError(
                f"YouTube video must be public. Private/unlisted videos not supported.{cache_context} "
                f"Original error: {error}",
            ) from error
        if "not found" in error_str or "unavailable" in error_str:
            raise APIError(
                f"Content not found or unavailable ({content_type}).{cache_context} "
                f"Original error: {error}",
            ) from error

        # Generic error with cache context
        raise APIError(
            f"Content generation failed for {content_type}{cache_context}: {error}",
        ) from error
