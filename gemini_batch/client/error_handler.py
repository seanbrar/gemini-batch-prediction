"""
Error handling for Gemini API generation requests
"""

from typing import Any, Optional

from ..exceptions import APIError


class GenerationErrorHandler:
    """Handles and enriches errors from Gemini API generation requests"""

    def handle_generation_error(
        self,
        error: Exception,
        response_schema: Optional[Any] = None,
        content_type: str = "unknown",
    ) -> None:
        """Transform API errors into informative error messages"""
        error_str = str(error).lower()

        # Structured output specific errors
        if response_schema and ("json" in error_str or "schema" in error_str):
            raise APIError(
                f"Structured output generation failed for {content_type} content. "
                f"Check your schema definition. Original error: {error}"
            ) from error

        # Content-specific errors
        if "quota" in error_str and "youtube" in error_str:
            raise APIError(
                f"YouTube quota exceeded. Free tier allows 8 hours/day. "
                f"Original error: {error}"
            ) from error
        elif "private" in error_str or "unlisted" in error_str:
            raise APIError(
                f"YouTube video must be public. Private/unlisted videos not supported. "
                f"Original error: {error}"
            ) from error
        elif "not found" in error_str or "unavailable" in error_str:
            raise APIError(
                f"Content not found or unavailable ({content_type}). "
                f"Original error: {error}"
            ) from error

        raise APIError(
            f"Content generation failed for {content_type}: {error}"
        ) from error
