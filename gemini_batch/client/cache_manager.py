"""
Cache lifecycle management for Gemini API context caching.

Handles explicit cache creation, tracking, and cleanup while integrating
with existing TokenCounter analysis and ConfigManager capabilities.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import time
from typing import Dict, List, Optional, Union

from google import genai
from google.genai import types

from ..config import ConfigManager


@dataclass
class CacheStrategy:
    """Cache strategy recommendation from analysis"""

    should_cache: bool
    strategy_type: str  # "explicit", "implicit", "none"
    estimated_tokens: int
    ttl_seconds: int = 3600  # Default 1 hour
    reason: str = ""


@dataclass
class CacheInfo:
    """Information about an active cache"""

    cache_name: str
    content_hash: str
    model: str
    created_at: datetime
    ttl_seconds: int
    token_count: int
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class CacheResult:
    """Result of cache operation"""

    success: bool
    cache_info: Optional[CacheInfo] = None
    cache_name: Optional[str] = None
    error: Optional[str] = None
    fallback_required: bool = False


@dataclass
class CacheMetrics:
    """Cache usage metrics for efficiency tracking"""

    total_caches: int = 0
    active_caches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cached_tokens: int = 0
    cache_creation_time: float = 0.0
    cache_savings_estimate: float = 0.0


class CacheManager:
    """
    Manages explicit cache lifecycle for Gemini API context caching.

    Integrates with existing TokenCounter for caching analysis and
    ConfigManager for model-specific capabilities.
    """

    def __init__(
        self,
        client: genai.Client,
        config_manager: ConfigManager,
        token_counter,
        default_ttl_seconds: int = 3600,
    ):
        """Initialize cache manager with existing components"""
        self.client = client
        self.config_manager = config_manager
        self.token_counter = token_counter
        self.default_ttl_seconds = default_ttl_seconds

        # Cache tracking
        self.active_caches: Dict[str, CacheInfo] = {}
        self.content_to_cache: Dict[str, str] = {}  # content_hash -> cache_name

        # Metrics
        self.metrics = CacheMetrics()

    def analyze_cache_strategy(
        self, model: str, content: Union[str, List], prefer_explicit: bool = True
    ) -> CacheStrategy:
        """
        Analyze content and determine optimal caching strategy.
        """
        try:
            # Use existing TokenCounter analysis
            cache_analysis = self.token_counter.estimate_for_caching(
                model, content, prefer_implicit=not prefer_explicit
            )

            if not cache_analysis["cacheable"]:
                return CacheStrategy(
                    should_cache=False,
                    strategy_type="none",
                    estimated_tokens=cache_analysis["tokens"],
                    reason="Below minimum token threshold or caching not supported",
                )

            strategy_type = cache_analysis["recommended_strategy"]
            estimated_tokens = cache_analysis["tokens"]

            # For explicit caching, calculate appropriate TTL
            if strategy_type == "explicit":
                ttl = self._calculate_optimal_ttl(estimated_tokens)
            else:
                ttl = 0  # Implicit caching doesn't use TTL

            return CacheStrategy(
                should_cache=True,
                strategy_type=strategy_type,
                estimated_tokens=estimated_tokens,
                ttl_seconds=ttl,
                reason=f"Content meets {strategy_type} caching requirements",
            )

        except Exception as e:
            return CacheStrategy(
                should_cache=False,
                strategy_type="none",
                estimated_tokens=0,
                reason=f"Cache analysis failed: {e}",
            )

    def get_or_create_cache(
        self,
        model: str,
        content_parts: List,
        strategy: CacheStrategy,
        system_instruction: Optional[str] = None,
    ) -> CacheResult:
        """
        Get existing cache or create new one for content.
        """
        if not strategy.should_cache or strategy.strategy_type != "explicit":
            return CacheResult(
                success=False,
                fallback_required=True,
                error="Content not suitable for explicit caching",
            )

        try:
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(
                content_parts, system_instruction
            )

            # Check for existing cache
            if content_hash in self.content_to_cache:
                cache_name = self.content_to_cache[content_hash]
                if cache_name in self.active_caches:
                    cache_info = self.active_caches[cache_name]

                    # Verify cache is still valid (not expired)
                    if self._is_cache_valid(cache_info):
                        # Update usage tracking
                        cache_info.usage_count += 1
                        cache_info.last_used = datetime.now(timezone.utc)
                        self.metrics.cache_hits += 1

                        return CacheResult(
                            success=True, cache_info=cache_info, cache_name=cache_name
                        )
                    else:
                        # Cache expired, clean it up
                        self._cleanup_cache(cache_name)

            # Create new cache
            cache_result = self._create_explicit_cache(
                model, content_parts, strategy, system_instruction, content_hash
            )

            if cache_result.success:
                self.metrics.cache_misses += 1

            return cache_result

        except Exception as e:
            return CacheResult(
                success=False,
                fallback_required=True,
                error=f"Cache operation failed: {e}",
            )

    def _create_explicit_cache(
        self,
        model: str,
        content_parts: List,
        strategy: CacheStrategy,
        system_instruction: Optional[str],
        content_hash: str,
    ) -> CacheResult:
        """Create new explicit cache"""
        start_time = time.time()

        try:
            # Prepare cache configuration
            cache_config = types.CreateCachedContentConfig(
                contents=content_parts, ttl=f"{strategy.ttl_seconds}s"
            )

            if system_instruction:
                cache_config.system_instruction = system_instruction

            # Create cache via API
            cache = self.client.caches.create(model=model, config=cache_config)

            # Track cache info
            cache_info = CacheInfo(
                cache_name=cache.name,
                content_hash=content_hash,
                model=model,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=strategy.ttl_seconds,
                token_count=strategy.estimated_tokens,
                usage_count=1,
                last_used=datetime.now(timezone.utc),
            )

            # Update tracking structures
            self.active_caches[cache.name] = cache_info
            self.content_to_cache[content_hash] = cache.name

            # Update metrics
            creation_time = time.time() - start_time
            self.metrics.total_caches += 1
            self.metrics.cache_creation_time += creation_time
            self.metrics.total_cached_tokens += strategy.estimated_tokens

            return CacheResult(
                success=True, cache_info=cache_info, cache_name=cache.name
            )

        except Exception as e:
            return CacheResult(
                success=False,
                fallback_required=True,
                error=f"Failed to create cache: {e}",
            )

    def _generate_content_hash(
        self, content_parts: List, system_instruction: Optional[str]
    ) -> str:
        """Generate hash for content deduplication"""
        # Create deterministic hash from content parts and system instruction
        hasher = hashlib.sha256()

        # Hash content parts
        for part in content_parts:
            if hasattr(part, "text"):
                hasher.update(part.text.encode("utf-8"))
            elif hasattr(part, "file_data") and hasattr(part.file_data, "file_uri"):
                hasher.update(part.file_data.file_uri.encode("utf-8"))
            else:
                hasher.update(str(part).encode("utf-8"))

        # Hash system instruction
        if system_instruction:
            hasher.update(system_instruction.encode("utf-8"))

        return hasher.hexdigest()

    def _calculate_optimal_ttl(self, token_count: int) -> int:
        """Calculate optimal TTL based on content size and usage patterns"""
        # Base TTL: 1 hour (API default)
        base_ttl = 3600

        # Larger content gets longer TTL (more expensive to recreate)
        if token_count > 100_000:
            return base_ttl * 4  # 4 hours for large content
        elif token_count > 50_000:
            return base_ttl * 2  # 2 hours for medium content
        else:
            return base_ttl  # 1 hour for smaller content

    def _is_cache_valid(self, cache_info: CacheInfo) -> bool:
        """Check if cache is still valid (not expired)"""
        now = datetime.now(timezone.utc)
        expiry_time = cache_info.created_at + timedelta(seconds=cache_info.ttl_seconds)
        return now < expiry_time

    def cleanup_expired_caches(self) -> int:
        """Clean up expired caches and return number cleaned"""
        cleaned_count = 0
        expired_caches = []

        # Identify expired caches
        for cache_name, cache_info in self.active_caches.items():
            if not self._is_cache_valid(cache_info):
                expired_caches.append(cache_name)

        # Clean up expired caches
        for cache_name in expired_caches:
            if self._cleanup_cache(cache_name):
                cleaned_count += 1

        return cleaned_count

    def _cleanup_cache(self, cache_name: str) -> bool:
        """Clean up a specific cache"""
        try:
            # Remove from API
            self.client.caches.delete(cache_name)

            # Remove from tracking
            if cache_name in self.active_caches:
                cache_info = self.active_caches[cache_name]
                content_hash = cache_info.content_hash

                del self.active_caches[cache_name]
                if content_hash in self.content_to_cache:
                    del self.content_to_cache[content_hash]

            return True

        except Exception:
            # Cache may already be deleted by API TTL
            # Clean up our tracking anyway
            if cache_name in self.active_caches:
                cache_info = self.active_caches[cache_name]
                content_hash = cache_info.content_hash

                del self.active_caches[cache_name]
                if content_hash in self.content_to_cache:
                    del self.content_to_cache[content_hash]

            return False

    def get_cache_metrics(self) -> CacheMetrics:
        """Get current cache usage metrics"""
        self.metrics.active_caches = len(self.active_caches)
        return self.metrics

    def list_active_caches(self) -> List[CacheInfo]:
        """List all active caches with their info"""
        return list(self.active_caches.values())
