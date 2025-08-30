import pytest

from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import Conversation
from gemini_batch.extensions.conversation_types import ConversationPolicy


@pytest.mark.unit
@pytest.mark.asyncio
async def test_basic_conversation_functionality():
    """Test basic conversation functionality that exists in the simplified implementation."""
    executor = create_executor()
    conv0 = Conversation.start(executor, sources=("alpha.txt",))

    # Test that conversation starts properly
    assert len(conv0.state.turns) == 0
    assert conv0.state.sources == ("alpha.txt",)

    # Test ask method exists and is callable
    conv1 = await conv0.ask("What is alpha?")
    assert len(conv0.state.turns) == 0  # original unchanged
    assert len(conv1.state.turns) == 1

    # Test with_policy method
    policy = ConversationPolicy(keep_last_n=5)
    conv_with_policy = conv1.with_policy(policy)
    assert conv_with_policy.state.policy.keep_last_n == 5

    # Test with_sources method with single source
    conv_with_sources = conv1.with_sources(["beta.txt"])
    assert "beta.txt" in conv_with_sources.state.sources
