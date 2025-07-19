"""Cache lifecycle management for Gemini API context caching.

Handles explicit cache creation, tracking, and cleanup while integrating
with existing TokenCounter analysis and ConfigManager capabilities.
"""

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from gemini_batch.client.content_processor import ContentProcessor
from gemini_batch.client.token_counter import TokenCounter
from gemini_batch.exceptions import APIError

from ..config import CachingRecommendation, ConfigManager
from .models import CacheAction, CacheStrategy, ExplicitCachePayload, PartsPayload

log = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about an active cache"""  # noqa: D415

    cache_name: str
    content_hash: str
    model: str
    created_at: datetime
    ttl_seconds: int
    token_count: int
    usage_count: int = 0
    last_used: datetime | None = None
    conversation_context_hash: str | None = None


@dataclass
class CacheResult:
    """Result of cache operation"""  # noqa: D415

    success: bool
    cache_info: CacheInfo | None = None
    cache_name: str | None = None
    error: str | None = None
    fallback_required: bool = False


@dataclass
class CacheMetrics:
    """Cache usage metrics for efficiency tracking"""  # noqa: D415

    total_caches: int = 0
    active_caches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cached_tokens: int = 0
    cache_creation_time: float = 0.0
    cache_savings_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format"""  # noqa: D415
        return {
            "total_caches": self.total_caches,
            "active_caches": self.active_caches,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_creation_time": self.cache_creation_time,
            "cache_savings_estimate": self.cache_savings_estimate,
        }


class CacheManager:
    """Manages explicit cache lifecycle for Gemini API context caching."""

    def __init__(  # noqa: D107
        self,
        client: genai.Client,
        config_manager: ConfigManager,
        token_counter: TokenCounter,
        default_ttl_seconds: int = 3600,
        content_processor: ContentProcessor = None,
    ):
        self.client = client
        self.config_manager = config_manager
        self.token_counter = token_counter
        self.default_ttl_seconds = default_ttl_seconds
        self.content_processor = content_processor

        # Cache tracking
        self.active_caches: dict[str, CacheInfo] = {}
        self.content_to_cache: dict[str, str] = {}
        self.metrics = CacheMetrics()

    def plan_generation(
        self,
        parts: list[types.Part],
        system_instruction: str | None = None,
    ) -> CacheAction:
        """Analyzes content and returns a plan for the client to execute."""
        # 1. Analyze content and determine strategy
        token_analysis = self.token_counter.estimate_for_caching(
            self.config_manager.model, parts
        )
        token_count = token_analysis["tokens"]
        cache_analysis = self.config_manager.can_use_caching(
            self.config_manager.model, token_count
        )

        if not cache_analysis.get("supported"):
            return CacheAction(CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts))

        strategy_name = cache_analysis["recommendation"]

        # 2. Simple strategy dispatch
        try:
            if strategy_name == CachingRecommendation.EXPLICIT:
                cache_info = self._ensure_explicit_cache(
                    parts, system_instruction, token_count
                )
                _, prompt_parts = self._prepare_cache_parts(parts)
                return CacheAction(
                    CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE,
                    ExplicitCachePayload(
                        cache_name=cache_info.cache_name, parts=prompt_parts
                    ),
                )

            if strategy_name == CachingRecommendation.IMPLICIT:
                optimized_parts = self.content_processor.optimize_for_implicit_cache(
                    parts
                )
                return CacheAction(
                    CacheStrategy.GENERATE_WITH_OPTIMIZED_PARTS,
                    PartsPayload(parts=optimized_parts),
                )

        except APIError as e:
            log.warning("Cache strategy failed, degrading to raw generation: %s", e)
            return CacheAction(CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts))

        return CacheAction(CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts))

    def _ensure_explicit_cache(
        self, parts: list[types.Part], system_instruction: str | None, token_count: int
    ) -> CacheInfo:
        """Ensure explicit cache exists, creating if necessary"""  # noqa: D415
        cacheable_parts, _ = self._prepare_cache_parts(parts)
        content_hash = self._hash_content(cacheable_parts, system_instruction, None)

        # Check for existing cache
        existing_cache = self._get_existing_cache(content_hash)
        if existing_cache and existing_cache.success:
            return existing_cache.cache_info

        # Create new cache
        ttl_seconds = self._calculate_ttl(token_count)
        cache_result = self._create_cache(
            self.config_manager.model,
            cacheable_parts,
            ttl_seconds,
            token_count,
            system_instruction,
            content_hash,
            None,
        )
        if not cache_result.success:
            raise APIError(f"Cache creation failed: {cache_result.error}")
        return cache_result.cache_info

    def _prepare_cache_parts(
        self, parts: list[types.Part]
    ) -> tuple[list[types.Part], list[types.Part]]:
        """Separate cacheable content from prompt parts with fallback handling"""  # noqa: D415
        cacheable_parts, prompt_parts = (
            self.content_processor.separate_cacheable_content(parts)
        )

        # Handle empty prompt_parts fallback
        if not prompt_parts:
            for part in parts:
                if part not in cacheable_parts:
                    prompt_parts = [part]
                    break
            if not prompt_parts and parts:
                prompt_parts = [parts[0]]  # Use first part as fallback

        return cacheable_parts, prompt_parts

    def _get_existing_cache(self, content_hash: str) -> CacheResult | None:
        """Check for existing valid cache"""  # noqa: D415
        if content_hash not in self.content_to_cache:
            return None

        cache_name = self.content_to_cache[content_hash]
        if cache_name not in self.active_caches:
            return None

        cache_info = self.active_caches[cache_name]
        log.debug(
            "Cache hit for content hash '%s' -> cache_name '%s'",
            content_hash[:8],
            cache_name,
        )

        # Check if cache is still valid
        if not self._is_cache_valid(cache_info):
            self._cleanup_cache(cache_name)
            return None

        # Update usage tracking
        cache_info.usage_count += 1
        cache_info.last_used = datetime.now(UTC)
        self.metrics.cache_hits += 1

        return CacheResult(success=True, cache_info=cache_info, cache_name=cache_name)

    def _create_cache(
        self,
        model: str,
        content_parts: list[types.Part],
        ttl_seconds: int,
        estimated_tokens: int,
        system_instruction: str | None,
        content_hash: str,
        conversation_context: str | None = None,
    ) -> CacheResult:
        """Create new explicit cache"""  # noqa: D415
        start_time = time.time()
        log.info(
            "Creating new explicit cache for model '%s' with TTL %ds.",
            model,
            ttl_seconds,
        )

        try:
            # Create cache configuration
            cache_config = types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=content_parts)],
                ttl=f"{ttl_seconds}s",
            )

            if system_instruction:
                cache_config.system_instruction = system_instruction

            # Create cache via API
            cache = self.client.caches.create(model=model, config=cache_config)

            # Create cache info
            cache_info = CacheInfo(
                cache_name=cache.name,
                content_hash=content_hash,
                model=model,
                created_at=datetime.now(UTC),
                ttl_seconds=ttl_seconds,
                token_count=estimated_tokens,
                usage_count=1,
                last_used=datetime.now(UTC),
                conversation_context_hash=self._hash_string(conversation_context)
                if conversation_context
                else None,
            )

            # Update tracking
            self.active_caches[cache.name] = cache_info
            self.content_to_cache[content_hash] = cache.name

            # Update metrics
            self.metrics.total_caches += 1
            self.metrics.cache_creation_time += time.time() - start_time
            self.metrics.total_cached_tokens += estimated_tokens
            self.metrics.cache_misses += 1

            return CacheResult(
                success=True,
                cache_info=cache_info,
                cache_name=cache.name,
            )

        except Exception as e:
            log.exception("Failed to create cache for model '%s'.", model)
            return CacheResult(
                success=False,
                fallback_required=True,
                error=f"Failed to create cache: {e}",
            )

    def _hash_content(
        self,
        content_parts: list[types.Part],
        system_instruction: str | None,
        conversation_context: str | None,
    ) -> str:
        """Generate hash for content deduplication"""  # noqa: D415
        hasher = hashlib.sha256()

        # Hash content parts
        for part in content_parts:
            if hasattr(part, "text") and part.text:
                hasher.update(part.text.encode("utf-8"))
            elif hasattr(part, "file_data") and hasattr(part.file_data, "file_uri"):
                hasher.update(part.file_data.file_uri.encode("utf-8"))
            else:
                hasher.update(str(part).encode("utf-8"))

        # Hash system instruction and conversation context
        for context in [system_instruction, conversation_context]:
            if context:
                hasher.update(f"||{context}||".encode())

        return hasher.hexdigest()

    def _hash_string(self, text: str) -> str:
        """Generate short hash for string content"""  # noqa: D415
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _calculate_ttl(self, token_count: int) -> int:
        """Calculate optimal TTL based on content size"""  # noqa: D415
        if token_count > 100_000:
            return self.default_ttl_seconds * 4  # 4 hours for large content
        if token_count > 50_000:
            return self.default_ttl_seconds * 2  # 2 hours for medium content
        return self.default_ttl_seconds  # 1 hour for smaller content

    def _is_cache_valid(self, cache_info: CacheInfo) -> bool:
        """Check if cache is within TTL"""  # noqa: D415
        return datetime.now(UTC) < cache_info.created_at + timedelta(
            seconds=cache_info.ttl_seconds,
        )

    def cleanup_expired_caches(self) -> int:
        """Cleanup expired caches and return count of cleaned caches."""
        cleaned_count = 0
        # Iterate over a copy of keys since we might modify the dict
        for cache_name in list(self.active_caches.keys()):
            cache_info = self.active_caches.get(cache_name)
            if cache_info and not self._is_cache_valid(cache_info):  # noqa: SIM102
                if self._cleanup_cache(cache_name):
                    cleaned_count += 1

        if cleaned_count > 0:
            log.info("Cleaned up %d expired caches.", cleaned_count)
        return cleaned_count

    def _cleanup_cache(self, cache_name: str) -> bool:
        """Cleanup a single cache entry and associated mappings."""
        try:
            # Remove from API (ignore errors - cache may be auto-expired)
            with suppress(Exception):
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
            return False

    def get_cache_metrics(self) -> CacheMetrics:
        """Get current cache usage metrics"""  # noqa: D415
        self.metrics.active_caches = len(self.active_caches)
        return self.metrics

    def list_active_caches(self) -> list[CacheInfo]:
        """List all active caches with their info"""  # noqa: D415
        return list(self.active_caches.values())
