"""Unit tests for result builder hint consumption and transform ordering."""

from typing import Any

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    FinalizedCommand,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Success,
    TextPart,
)
from gemini_batch.pipeline.hints import ResultHint
from gemini_batch.pipeline.result_builder import ResultBuilder
from gemini_batch.pipeline.results.transforms import (
    json_array_transform,
    simple_text_transform,
)

pytestmark = pytest.mark.unit


class TestResultBuilderHints:
    """Test result builder hint consumption and transform bias."""

    @pytest.fixture
    def basic_finalized_command(self):
        """Create a basic finalized command for testing."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        # Mock response that could match JSON array transform
        mock_response = '["answer1", "answer2"]'

        return FinalizedCommand(planned=planned, raw_api_response=mock_response)

    @pytest.fixture
    def builder_with_json_transforms(self):
        """Create result builder with json_array and simple_text transforms."""
        transforms = [
            json_array_transform(),
            simple_text_transform(),
        ]
        return ResultBuilder(transforms=tuple(transforms), enable_diagnostics=True)

    @pytest.mark.asyncio
    async def test_no_hints_uses_default_transform_order(
        self, builder_with_json_transforms, basic_finalized_command
    ):
        """Without hints, should use default priority-based transform order."""
        result = await builder_with_json_transforms.handle(basic_finalized_command)

        assert isinstance(result, Success)
        envelope = result.value
        assert envelope["success"] is True

        # JSON array transform has priority 90, simple_text has priority 50
        # So json_array should be tried first and succeed
        assert envelope["extraction_method"] == "json_array"
        # Should extract JSON array (may be normalized/padded by result builder)
        assert isinstance(envelope["answers"], list)
        assert len(envelope["answers"]) >= 1

    @pytest.mark.asyncio
    async def test_result_hint_biases_json_array_to_front(
        self, builder_with_json_transforms
    ):
        """ResultHint with prefer_json_array should bias json_array transform first."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(ResultHint(prefer_json_array=True),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        # Response that json_array transform can handle
        finalized = FinalizedCommand(
            planned=planned, raw_api_response='["biased1", "biased2"]'
        )

        result = await builder_with_json_transforms.handle(finalized)

        assert isinstance(result, Success)
        envelope = result.value
        assert envelope["success"] is True
        assert envelope["extraction_method"] == "json_array"
        # Should have hint metadata in metrics
        assert (
            envelope.get("metrics", {}).get("hints", {}).get("prefer_json_array")
            is True
        )

    @pytest.mark.asyncio
    async def test_result_hint_fallback_still_works(self, builder_with_json_transforms):
        """ResultHint should not break Tier-2 fallback when JSON parsing fails."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(ResultHint(prefer_json_array=True),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        # Response that isn't valid JSON
        finalized = FinalizedCommand(
            planned=planned, raw_api_response="This is plain text, not JSON"
        )

        result = await builder_with_json_transforms.handle(finalized)

        assert isinstance(result, Success)
        envelope = result.value
        assert envelope["success"] is True  # Should still succeed via fallback
        # Should fall back to simple_text or minimal projection
        assert envelope["extraction_method"] in ["simple_text", "minimal_text"]

    @pytest.mark.asyncio
    async def test_result_hint_false_preserves_normal_order(
        self,
        builder_with_json_transforms,
        basic_finalized_command,  # noqa: ARG002
    ):
        """ResultHint with prefer_json_array=False should not change ordering."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(ResultHint(prefer_json_array=False),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        finalized = FinalizedCommand(
            planned=planned, raw_api_response='["normal1", "normal2"]'
        )

        result = await builder_with_json_transforms.handle(finalized)

        assert isinstance(result, Success)
        envelope = result.value
        assert envelope["success"] is True
        assert envelope["extraction_method"] == "json_array"
        # Should NOT have hint metadata for false preference
        assert (
            envelope.get("metrics", {}).get("hints", {}).get("prefer_json_array")
            is not True
        )

    @pytest.mark.asyncio
    async def test_sorted_transforms_for_preserves_stability(
        self, builder_with_json_transforms
    ):
        """_sorted_transforms_for should maintain stable ordering when no biasing."""
        # Get default order
        default_order = builder_with_json_transforms._sorted_transforms

        # Get order with no hints
        no_hints_order = builder_with_json_transforms._sorted_transforms_for(None)

        # Get order with empty hints
        empty_hints_order = builder_with_json_transforms._sorted_transforms_for(())

        # All should be identical
        assert default_order == no_hints_order == empty_hints_order

    @pytest.mark.asyncio
    async def test_sorted_transforms_for_bubbles_json_array_when_hinted(
        self,
        builder_with_json_transforms,  # noqa: ARG002
    ):
        """_sorted_transforms_for should move json_array transform to front when hinted."""
        # Create a builder with mixed priority transforms to test reordering
        from gemini_batch.pipeline.results.extraction import TransformSpec

        def dummy_matcher(response: Any) -> bool:  # noqa: ARG001
            return True

        def dummy_extractor(response: Any, config: Any) -> dict[str, Any]:  # noqa: ARG001
            return {"answers": ["dummy"], "confidence": 0.5}

        # High priority transform that would normally come first
        high_priority_transform = TransformSpec(
            name="high_priority",
            matcher=dummy_matcher,
            extractor=dummy_extractor,
            priority=100,
        )

        builder = ResultBuilder(
            transforms=(
                high_priority_transform,
                json_array_transform(),
                simple_text_transform(),
            )
        )

        # Without hint - high priority should be first
        normal_order = builder._sorted_transforms_for(None)
        assert normal_order[0].name == "high_priority"

        # With hint - json_array should be bubbled to front
        hint_order = builder._sorted_transforms_for(
            (ResultHint(prefer_json_array=True),)
        )
        assert hint_order[0].name == "json_array"

    @pytest.mark.asyncio
    async def test_multiple_result_hints_only_first_preference_matters(
        self, builder_with_json_transforms
    ):
        """Multiple ResultHints should work but only meaningful preferences applied."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(
                ResultHint(prefer_json_array=True),
                ResultHint(prefer_json_array=False),  # Second hint ignored
            ),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        finalized = FinalizedCommand(
            planned=planned, raw_api_response='["multi1", "multi2"]'
        )

        result = await builder_with_json_transforms.handle(finalized)

        assert isinstance(result, Success)
        envelope = result.value
        # Should still bias toward json_array (first hint wins)
        assert (
            envelope.get("metrics", {}).get("hints", {}).get("prefer_json_array")
            is True
        )

    @pytest.mark.asyncio
    async def test_result_hint_with_unknown_transform_name_safe(
        self,
        builder_with_json_transforms,  # noqa: ARG002
    ):
        """ResultHint should safely handle case where json_array transform doesn't exist."""
        # Create builder without json_array transform
        builder_no_json = ResultBuilder(transforms=(simple_text_transform(),))

        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(ResultHint(prefer_json_array=True),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(calls=(call,))
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        finalized = FinalizedCommand(
            planned=planned, raw_api_response="plain text response"
        )

        result = await builder_no_json.handle(finalized)

        # Should still succeed - no json_array transform to bubble, so normal order
        assert isinstance(result, Success)
        envelope = result.value
        assert envelope["success"] is True
