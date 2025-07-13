"""
Cache lifecycle management for Gemini API context caching.

Handles explicit cache creation, tracking, and cleanup while integrating
with existing TokenCounter analysis and ConfigManager capabilities.
"""

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

from gemini_batch.client.content_processor import ContentProcessor
from gemini_batch.client.token_counter import TokenCounter
from gemini_batch.exceptions import APIError

from ..config import ConfigManager
from .models import CacheAction, CacheStrategy, ExplicitCachePayload, PartsPayload

log = logging.getLogger(__name__)


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
    conversation_context_hash: Optional[str] = None


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format"""
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

    def __init__(
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
        self.active_caches: Dict[str, CacheInfo] = {}
        self.content_to_cache: Dict[str, str] = {}
        self.metrics = CacheMetrics()

    def analyze_cache_strategy(
        self,
        model: str,
        content: Union[str, List],
        prefer_explicit: bool = True,
        conversation_context: Optional[str] = None,
    ) -> CacheStrategy:
        """Analyze content and determine optimal caching strategy."""
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

            # Calculate total tokens including conversation context
            total_tokens = cache_analysis["tokens"]
            if conversation_context:
                context_analysis = self.token_counter.estimate_for_caching(
                    model, conversation_context, prefer_implicit=False
                )
                total_tokens += context_analysis["tokens"]

            return CacheStrategy(
                should_cache=True,
                strategy_type=cache_analysis["recommended_strategy"],
                estimated_tokens=total_tokens,
                ttl_seconds=self._calculate_ttl(total_tokens),
                reason=f"Content meets {cache_analysis['recommended_strategy']} caching requirements",
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
        conversation_context: Optional[str] = None,
    ) -> CacheResult:
        """Get existing cache or create new one for content."""
        if not strategy.should_cache or strategy.strategy_type != "explicit":
            return CacheResult(
                success=False,
                fallback_required=True,
                error="Content not suitable for explicit caching",
            )

        try:
            # Generate content hash for deduplication
            content_hash = self._hash_content(
                content_parts, system_instruction, conversation_context
            )

            # Check for existing valid cache
            existing_cache = self._get_existing_cache(content_hash)
            if existing_cache:
                return existing_cache

            # Create new cache
            return self._create_cache(
                model,
                content_parts,
                strategy,
                system_instruction,
                content_hash,
                conversation_context,
            )

        except Exception as e:
            return CacheResult(
                success=False,
                fallback_required=True,
                error=f"Cache operation failed: {e}",
            )

    def create_or_get(
        self,
        parts: List[types.Part],
        model: str,
        strategy: str = "explicit",
        system_instruction: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> CacheInfo:
        """Get existing cache or create new one for content parts."""
        # Analyze cache strategy
        cache_strategy = self.analyze_cache_strategy(
            model,
            parts,
            prefer_explicit=(strategy == "explicit"),
            conversation_context=conversation_context,
        )

        # Get or create cache
        result = self.get_or_create_cache(
            model, parts, cache_strategy, system_instruction, conversation_context
        )

        if not result.success:
            raise APIError(f"Failed to create or get cache: {result.error}")

        return result.cache_info

    def plan_generation(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
    ) -> CacheAction:
        """
        Analyzes content and returns a plan for the client to execute.

        This is the core of the Planner-Executor pattern. The CacheManager acts as a Planner,
        analyzing the request and returning a simple, descriptive action plan. The GeminiClient
        then acts as an Executor, reading the plan and performing the corresponding action.
        """
        # 1. Analyze content and determine the appropriate caching strategy.
        token_count = self.token_counter.estimate_for_caching(self.config_manager.model, parts)
        cache_analysis = self.config_manager.can_use_caching(
            self.config_manager.model, token_count
        )

        if not cache_analysis.get("supported"):
            return CacheAction(
                CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts)
            )

        strategy_name = cache_analysis["recommendation"]
        log.debug("Cache strategy determined: %s", strategy_name)

        # 2. Create the appropriate action plan based on the strategy.
        try:
            if strategy_name == "explicit":
                # Perform cache creation/retrieval here (the Planner's only side effect)
                cache_info = self.create_or_get(
                    parts,
                    self.config_manager.model,
                    strategy="explicit",
                    system_instruction=system_instruction,
                )  # returns CacheInfo
                _, prompt_parts = self.content_processor.separate_cacheable_content(
                    parts
                )
                return CacheAction(
                    CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE,
                    ExplicitCachePayload(
                        cache_name=cache_info.cache_name, parts=prompt_parts
                    ),
                )
            elif strategy_name == "implicit":
                optimized_parts = self.content_processor.optimize_for_implicit_cache(
                    parts
                )
                return CacheAction(
                    CacheStrategy.GENERATE_WITH_OPTIMIZED_PARTS,
                    PartsPayload(parts=optimized_parts),
                )

        except APIError as e:
            # Graceful Degradation: If cache creation fails, fall back to a raw generation plan.
            log.warning(
                "Explicit cache creation failed for model '%s', degrading to raw generation. Reason: %s",
                self.config_manager.model,
                e,
            )
            return CacheAction(
                CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts)
            )

        # 3. Default fallback for any strategy that isn't explicitly handled (e.g., "none").
        return CacheAction(CacheStrategy.GENERATE_RAW, PartsPayload(parts=parts))

    def _get_existing_cache(self, content_hash: str) -> Optional[CacheResult]:
        """Check for existing valid cache"""
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
        cache_info.last_used = datetime.now(timezone.utc)
        self.metrics.cache_hits += 1

        return CacheResult(success=True, cache_info=cache_info, cache_name=cache_name)

    def _create_cache(
        self,
        model: str,
        content_parts: List,
        strategy: CacheStrategy,
        system_instruction: Optional[str],
        content_hash: str,
        conversation_context: Optional[str],
    ) -> CacheResult:
        """Create new explicit cache"""
        start_time = time.time()
        log.info(
            "Creating new explicit cache for model '%s' with TTL %ds.",
            model,
            strategy.ttl_seconds,
        )

        try:
            # Create cache configuration
            cache_config = types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=content_parts)],
                ttl=f"{strategy.ttl_seconds}s",
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
                created_at=datetime.now(timezone.utc),
                ttl_seconds=strategy.ttl_seconds,
                token_count=strategy.estimated_tokens,
                usage_count=1,
                last_used=datetime.now(timezone.utc),
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
            self.metrics.total_cached_tokens += strategy.estimated_tokens
            self.metrics.cache_misses += 1

            return CacheResult(
                success=True, cache_info=cache_info, cache_name=cache.name
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
        content_parts: List,
        system_instruction: Optional[str],
        conversation_context: Optional[str],
    ) -> str:
        """Generate hash for content deduplication"""
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
        """Generate short hash for string content"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _calculate_ttl(self, token_count: int) -> int:
        """Calculate optimal TTL based on content size"""
        if token_count > 100_000:
            return self.default_ttl_seconds * 4  # 4 hours for large content
        elif token_count > 50_000:
            return self.default_ttl_seconds * 2  # 2 hours for medium content
        else:
            return self.default_ttl_seconds  # 1 hour for smaller content

    def _is_cache_valid(self, cache_info: CacheInfo) -> bool:
        """Check if cache is within TTL"""
        return datetime.now(timezone.utc) < cache_info.created_at + timedelta(
            seconds=cache_info.ttl_seconds
        )

    def cleanup_expired_caches(self) -> int:
        """Cleanup expired caches and return count of cleaned caches."""
        cleaned_count = 0
        # Iterate over a copy of keys since we might modify the dict
        for cache_name in list(self.active_caches.keys()):
            cache_info = self.active_caches.get(cache_name)
            if cache_info and not self._is_cache_valid(cache_info):
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
        """Get current cache usage metrics"""
        self.metrics.active_caches = len(self.active_caches)
        return self.metrics

    def list_active_caches(self) -> List[CacheInfo]:
        """List all active caches with their info"""
        return list(self.active_caches.values())
