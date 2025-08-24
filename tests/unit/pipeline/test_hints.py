"""Unit tests for hint capsules type contracts and immutability."""

import pytest

from gemini_batch.pipeline.hints import (
    CacheHint,
    EstimationOverrideHint,
    ExecutionCacheName,
    Hint,
    ResultHint,
)

pytestmark = pytest.mark.unit


class TestHintTypeContracts:
    """Verify hint types follow architectural contracts."""

    def test_cache_hint_is_frozen_and_immutable(self):
        """CacheHint should be frozen dataclass with immutable behavior."""
        hint = CacheHint("test-key")
        assert hint.deterministic_key == "test-key"
        assert hint.artifacts == ()
        assert hint.ttl_seconds is None
        assert hint.reuse_only is False

        # Should be frozen
        with pytest.raises(AttributeError):
            hint.deterministic_key = "new-key"  # type: ignore

    def test_cache_hint_with_all_fields(self):
        """CacheHint should support all optional fields."""
        hint = CacheHint(
            deterministic_key="conv:123",
            artifacts=("artifact1", "artifact2"),
            ttl_seconds=3600,
            reuse_only=True,
        )
        assert hint.deterministic_key == "conv:123"
        assert hint.artifacts == ("artifact1", "artifact2")
        assert hint.ttl_seconds == 3600
        assert hint.reuse_only is True

    def test_estimation_override_hint_is_frozen_and_immutable(self):
        """EstimationOverrideHint should be frozen with defaults."""
        hint = EstimationOverrideHint()
        assert hint.widen_max_factor == 1.0
        assert hint.clamp_max_tokens is None

        # Should be frozen
        with pytest.raises(AttributeError):
            hint.widen_max_factor = 2.0  # type: ignore

    def test_estimation_override_hint_with_overrides(self):
        """EstimationOverrideHint should support conservative adjustments."""
        hint = EstimationOverrideHint(widen_max_factor=1.5, clamp_max_tokens=16000)
        assert hint.widen_max_factor == 1.5
        assert hint.clamp_max_tokens == 16000

    def test_result_hint_is_frozen_and_immutable(self):
        """ResultHint should be frozen with defaults."""
        hint = ResultHint()
        assert hint.prefer_json_array is False

        # Should be frozen
        with pytest.raises(AttributeError):
            hint.prefer_json_array = True  # type: ignore

    def test_result_hint_with_json_preference(self):
        """ResultHint should support JSON array preference."""
        hint = ResultHint(prefer_json_array=True)
        assert hint.prefer_json_array is True

    def test_execution_cache_name_is_frozen_and_immutable(self):
        """ExecutionCacheName should be frozen with required cache name."""
        hint = ExecutionCacheName("cache-123")
        assert hint.cache_name == "cache-123"

        # Should be frozen
        with pytest.raises(AttributeError):
            hint.cache_name = "new-cache"  # type: ignore

    def test_hint_union_type_includes_all_hint_types(self):
        """Hint union should include all defined hint types."""
        cache = CacheHint("key")
        estimation = EstimationOverrideHint()
        result = ResultHint()
        execution = ExecutionCacheName("cache")

        # Should all be valid Hint types
        hints: list[Hint] = [cache, estimation, result, execution]
        assert len(hints) == 4

    def test_hints_are_hashable_for_deduplication(self):
        """All hint types should be hashable for use in sets/dicts."""
        cache1 = CacheHint("key1")
        cache2 = CacheHint("key1")
        cache3 = CacheHint("key2")

        hint_set = {cache1, cache2, cache3}
        assert len(hint_set) == 2  # cache1 and cache2 should be equal

    def test_hints_have_meaningful_equality(self):
        """Hints with same values should be equal."""
        assert CacheHint("key") == CacheHint("key")
        assert CacheHint("key1") != CacheHint("key2")

        assert EstimationOverrideHint(widen_max_factor=2.0) == EstimationOverrideHint(
            widen_max_factor=2.0
        )
        assert EstimationOverrideHint(widen_max_factor=1.0) != EstimationOverrideHint(
            widen_max_factor=2.0
        )

        assert ResultHint(prefer_json_array=True) == ResultHint(prefer_json_array=True)
        assert ResultHint(prefer_json_array=False) != ResultHint(prefer_json_array=True)

        assert ExecutionCacheName("cache1") == ExecutionCacheName("cache1")
        assert ExecutionCacheName("cache1") != ExecutionCacheName("cache2")

    def test_hints_have_meaningful_string_representation(self):
        """Hints should have useful string representations for debugging."""
        cache = CacheHint("test-key", artifacts=("a1",), ttl_seconds=3600)
        assert "test-key" in str(cache)
        assert "3600" in str(cache)

        estimation = EstimationOverrideHint(
            widen_max_factor=1.5, clamp_max_tokens=10000
        )
        assert "1.5" in str(estimation)
        assert "10000" in str(estimation)

        result = ResultHint(prefer_json_array=True)
        assert "True" in str(result)

        execution = ExecutionCacheName("my-cache")
        assert "my-cache" in str(execution)
