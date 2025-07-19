"""
Error handling for Gemini API generation requests
"""  # noqa: D200, D212, D415

from typing import Any, Optional

from ..exceptions import APIError  # noqa: TID252


class GenerationErrorHandler:
    """Handles and enriches errors from Gemini API generation requests"""  # noqa: D415

    def handle_generation_error(
        self,
        error: Exception,
        response_schema: Optional[Any] = None,  # noqa: ANN401, UP045
        content_type: str = "unknown",
        cache_enabled: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Transform API errors into informative error messages"""  # noqa: D415
        error_str = str(error).lower()

        # Cache-specific error handling
        if cache_enabled and any(
            cache_term in error_str for cache_term in ["cache", "cached_content"]
        ):
            if "not found" in error_str or "expired" in error_str:
                raise APIError(  # noqa: TRY003
                    f"Cache expired or not found. Falling back to non-cached generation. "  # noqa: E501, EM102
                    f"Original error: {error}"  # noqa: COM812
                ) from error
            elif "ttl" in error_str:  # noqa: RET506
                raise APIError(  # noqa: TRY003
                    f"Cache TTL configuration error. Check cache duration settings. "  # noqa: EM102
                    f"Original error: {error}"  # noqa: COM812
                ) from error
            elif "cached_content" in error_str:
                raise APIError(  # noqa: TRY003
                    f"Cache content validation failed. Content may be too large or malformed. "  # noqa: E501, EM102
                    f"Original error: {error}"  # noqa: COM812
                ) from error

        # Structured output specific errors
        if response_schema and ("json" in error_str or "schema" in error_str):
            cache_context = " (with caching enabled)" if cache_enabled else ""
            raise APIError(  # noqa: TRY003
                f"Structured output generation failed for {content_type} content{cache_context}. "  # noqa: E501, EM102
                f"Check your schema definition. Original error: {error}"  # noqa: COM812
            ) from error

        # Content-specific errors with cache context
        cache_context = (
            " Note: Caching was enabled for this request." if cache_enabled else ""
        )

        if "quota" in error_str and "youtube" in error_str:
            raise APIError(  # noqa: TRY003
                f"YouTube quota exceeded. Free tier allows 8 hours/day.{cache_context} "  # noqa: EM102
                f"Original error: {error}"  # noqa: COM812
            ) from error
        elif "private" in error_str or "unlisted" in error_str:  # noqa: RET506
            raise APIError(  # noqa: TRY003
                f"YouTube video must be public. Private/unlisted videos not supported.{cache_context} "  # noqa: E501, EM102
                f"Original error: {error}"  # noqa: COM812
            ) from error
        elif "not found" in error_str or "unavailable" in error_str:
            raise APIError(  # noqa: TRY003
                f"Content not found or unavailable ({content_type}).{cache_context} "  # noqa: EM102
                f"Original error: {error}"  # noqa: COM812
            ) from error

        # Generic error with cache context
        raise APIError(  # noqa: TRY003
            f"Content generation failed for {content_type}{cache_context}: {error}"  # noqa: COM812, EM102
        ) from error
