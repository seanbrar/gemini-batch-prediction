import logging
from typing import Any

from ..config import ConfigManager

log = logging.getLogger(__name__)


class TokenCounter:
    """Conservative token estimation and caching decision support for Gemini API.

    This class addresses observed inaccuracies in Gemini's count_tokens() API by applying
    content-type-specific adjustments based on empirical testing. Preliminary analysis indicates:

    - Images: Significant undercounting (approximately 5x actual usage)
    - Videos: Moderate overcounting (approximately 18% above actual)
    - Text/PDFs: High accuracy with minor consistent undercount
    - Mixed content: Behavior influenced by the most problematic media type

    The class provides conservative estimates suitable for caching decisions and
    integrates with the configuration system to recommend optimal caching strategies.
    """

    def __init__(self, client, config_manager: ConfigManager):
        self.client = client
        self.config_manager = config_manager

    def estimate_for_caching(
        self,
        model: str,
        content: Any,
        prefer_implicit: bool = True,
    ) -> dict[str, Any]:
        """Get comprehensive token estimate optimized for caching decisions.

        Applies content-type-specific adjustments to raw count_tokens() results
        to provide more accurate estimates, especially near caching thresholds.

        Args:
            model: Gemini model name
            content: Content to analyze (text, list of parts, etc.)
            prefer_implicit: Whether to prefer implicit over explicit caching

        Returns:
            Dictionary containing:
            - tokens: Adjusted token count
            - base_tokens: Raw API result
            - safety_margin_applied: Difference between adjusted and base
            - content_type: Detected content type
            - caching: Full caching analysis from config system
            - cacheable: Whether content is cacheable with any strategy
            - recommended_strategy: Best caching approach for this content
        """
        try:
            # Get base count from API
            result = self.client.models.count_tokens(model=model, contents=content)
            base_tokens = result.total_tokens

            # Apply content-type-specific safety margins
            adjusted_tokens = self._apply_smart_safety_margin(base_tokens, content)

            # Handle critical edge cases near thresholds
            adjusted_tokens = self._handle_threshold_edge_cases(
                adjusted_tokens,
                base_tokens,
                model,
                content,
            )

            # Get caching analysis from config system
            caching_analysis = self.config_manager.can_use_caching(
                model,
                adjusted_tokens,
                prefer_implicit,
            )

            return {
                "tokens": adjusted_tokens,
                "base_tokens": base_tokens,
                "safety_margin_applied": adjusted_tokens - base_tokens,
                "content_type": self._detect_content_type(content),
                "caching": caching_analysis,
                "cacheable": caching_analysis["implicit"]
                or caching_analysis["explicit"],
                "recommended_strategy": caching_analysis["recommendation"].value,
            }

        except Exception as e:
            # Fallback to rough estimation if API call fails
            estimated = self._rough_estimate(content)
            caching_analysis = self.config_manager.can_use_caching(
                model,
                estimated,
                prefer_implicit,
            )

            return {
                "tokens": estimated,
                "base_tokens": estimated,
                "safety_margin_applied": 0,
                "content_type": self._detect_content_type(content),
                "caching": caching_analysis,
                "cacheable": caching_analysis["implicit"]
                or caching_analysis["explicit"],
                "recommended_strategy": caching_analysis["recommendation"].value,
                "estimation_error": str(e),
            }

    def _apply_smart_safety_margin(self, base_tokens: int, content: Any) -> int:
        """Apply content-type-specific safety margins to address observed API discrepancies.

        Empirical testing indicates different content types exhibit distinct patterns:
        - Text/PDF: Minor consistent undercount (typically ~4 tokens)
        - Video: Moderate overcount (approximately 18% above actual usage)
        - Images: Significant undercount (approximately 5x actual usage)
        - Mixed: Behavior influenced by most problematic component

        Args:
            base_tokens: Raw token count from count_tokens() API
            content: The content being analyzed

        Returns:
            Adjusted token count with appropriate safety margin
        """
        content_type = self._detect_content_type(content)

        # For mixed content, analyze composition to determine dominant behavior
        if content_type == "mixed":
            has_video, has_image = self._analyze_mixed_content(content)

            if has_video and not has_image:
                # Video-dominated: compensate for observed overcount pattern
                return int(base_tokens * 0.85)  # 15% reduction
            if has_image:
                # Image-dominated: apply correction for observed undercount pattern
                return int(base_tokens * 4.0)  # Conservative 4x multiplier
            # Fallback for unexpected mixed content
            return base_tokens + 10

        if "video" in content_type:
            # Pure video: reduce to compensate for observed overcount pattern
            return int(base_tokens * 0.85)
        if "image" in content_type:
            # Images require larger multiplier due to observed undercount pattern
            return int(base_tokens * 6.0)  # Conservative approach beyond observed ratio
        # Text-only: add buffer for observed minor undercount pattern
        return base_tokens + 10

    def _analyze_mixed_content(self, content: Any) -> tuple[bool, bool]:
        """Analyze mixed content to identify media types present.

        Returns:
            Tuple of (has_video, has_image) booleans
        """
        has_video = False
        has_image = False

        if isinstance(content, list):
            for item in content:
                if hasattr(item, "mime_type") and item.mime_type:
                    if "video" in item.mime_type:
                        has_video = True
                    elif "image" in item.mime_type:
                        has_image = True
                elif hasattr(item, "file_uri"):
                    uri = str(item.file_uri).lower()
                    if any(ext in uri for ext in [".mp4", ".mov", ".avi", ".webm"]):
                        has_video = True
                    elif any(
                        ext in uri for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        has_image = True

        return has_video, has_image

    def _detect_content_type(self, content: Any) -> str:
        """Detect the primary content type to inform safety margin selection.

        For mixed content, prioritizes the most challenging type for accurate counting:
        images > video > text (based on observed counting accuracy patterns)

        Returns:
            Content type string: 'text', 'image', 'video', 'mixed', or 'unknown'
        """
        if isinstance(content, str):
            return "text"
        if isinstance(content, list):
            has_video = False
            has_image = False
            has_text = False

            for item in content:
                if hasattr(item, "mime_type") and item.mime_type:
                    if "video" in item.mime_type:
                        has_video = True
                    elif "image" in item.mime_type:
                        has_image = True
                elif hasattr(item, "file_uri"):
                    # Handle uploaded files by extension
                    uri = str(item.file_uri).lower()
                    if any(ext in uri for ext in [".mp4", ".mov", ".avi", ".webm"]):
                        has_video = True
                    elif any(
                        ext in uri for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                    ):
                        has_image = True
                else:
                    has_text = True

            # Priority order based on observed counting challenges: images > video > text
            if has_image:
                return "image" if not (has_video or has_text) else "mixed"
            if has_video:
                return "video" if not has_text else "mixed"
            return "text"
        return "unknown"

    def _rough_estimate(self, content: Any) -> int:
        """Fallback estimation when count_tokens() API is unavailable.

        Uses conservative estimates based on content type and observed patterns.
        """
        if isinstance(content, str):
            # Text estimation: approximately 4 characters per token with buffer for undercount
            return max(len(content) // 4 + 10, 50)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, str):
                    total += len(item) // 4 + 10
                elif hasattr(item, "mime_type"):
                    if "video" in item.mime_type:
                        total += 2000  # Conservative video estimate
                    elif "image" in item.mime_type:
                        total += (
                            1500  # Higher estimate accounting for undercount patterns
                        )
                    else:
                        total += 800  # Other file types
                elif hasattr(item, "file_uri"):
                    # Estimate based on file extension
                    uri = str(item.file_uri).lower()
                    if any(ext in uri for ext in [".mp4", ".mov", ".avi"]):
                        total += 2000
                    elif any(ext in uri for ext in [".jpg", ".png", ".gif"]):
                        total += 1500
                    else:
                        total += 800
                else:
                    total += 500
            return max(total, 100)
        return 1000  # Conservative default

    def get_caching_thresholds(self, model: str) -> dict[str, int | None]:
        """Get caching thresholds for a model - delegates to config system."""
        return self.config_manager.get_caching_thresholds(model)

    def _handle_threshold_edge_cases(
        self,
        adjusted_tokens: int,
        base_tokens: int,
        model: str,
        content: Any,
    ) -> int:
        """Apply additional safety buffer for content near caching thresholds.

        Addresses cases where count_tokens() may predict content as non-cacheable
        but actual usage crosses the threshold. This is particularly important for text
        content where the API shows generally high accuracy but small differences matter.

        Args:
            adjusted_tokens: Token count after initial safety margins
            base_tokens: Original count_tokens() result
            model: Model name for threshold lookup
            content: Content being analyzed

        Returns:
            Final adjusted token count with threshold safety buffer if needed
        """
        content_type = self._detect_content_type(content)

        # Focus on text content where threshold edge cases are most critical
        if content_type != "text":
            return adjusted_tokens

        # Get thresholds for this model
        thresholds = self.config_manager.get_caching_thresholds(model)

        # Define safety zone around each threshold
        danger_zone = 20

        # Check each threshold value (not the keys)
        for threshold_value in thresholds.values():
            if threshold_value is None:
                continue
            # Only compare if threshold_value is an int (or can be cast to int)
            try:
                threshold_int = int(threshold_value)
            except (TypeError, ValueError):
                continue  # Skip non-integer thresholds

            # Apply safety margin for content near threshold
            if (threshold_int - danger_zone) <= adjusted_tokens < threshold_int:
                # Add extra buffer for content near threshold
                adjusted_tokens = threshold_int + danger_zone
                log.debug(
                    "Applied threshold safety margin: %d -> %d (threshold: %d)",
                    base_tokens,
                    adjusted_tokens,
                    threshold_int,
                )
                break

        return adjusted_tokens
