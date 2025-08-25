import pytest

from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import Conversation, Flow


@pytest.mark.unit
@pytest.mark.asyncio
async def test_facade_is_immutable_and_chainable():
    executor = create_executor()
    conv0 = Conversation.start(executor, sources=("alpha.txt",))

    # Ask one question -> new conversation
    conv1 = await conv0.ask("What is alpha?")
    assert len(conv0.state.turns) == 0  # original unchanged
    assert len(conv1.state.turns) == 1
    assert conv1.state.last and isinstance(conv1.state.last.assistant, str)

    # ask_many appends two turns and returns answers
    conv2, answers = await conv1.ask_many("Define A.", "List A facts.")
    assert len(answers) == 2
    assert len(conv2.state.turns) == 3
    assert conv2.state.last and "echo:" in conv2.state.last.assistant


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ask_batch_sequential_appends_one_turn_per_prompt():
    executor = create_executor()
    conv = Conversation.start(executor, sources=("alpha.txt",))

    prompts = ("Q1", "Q2", "Q3")
    conv2, answers, metrics = await conv.ask_batch(prompts, vectorized=False)
    assert len(answers) == len(prompts)
    assert len(conv2.state.turns) == len(prompts)
    assert isinstance(metrics.totals, dict)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_flow_runs_linear_sequence():
    executor = create_executor()
    conv = Conversation.start(executor, sources=("alpha.txt",))

    flow = Flow().ask("Outline").ask("Write intro")
    conv2, answers, metrics = await conv.run(flow)
    assert len(answers) == 2
    assert len(conv2.state.turns) == 2
    assert "duration" in "".join(metrics.totals.keys()) or isinstance(
        metrics.totals, dict
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ask_batch_vectorized_creates_synthetic_turn():
    executor = create_executor()
    conv = Conversation.start(executor, sources=("alpha.txt",))

    prompts = ("Provide an outline.", "Write a title.")
    conv2, answers, metrics = await conv.ask_batch(prompts, vectorized=True)
    # Answers tuple size is preserved even if core joins prompts
    assert len(answers) == len(prompts)
    # Only one synthetic turn appended
    assert len(conv2.state.turns) == 1
    # Synthetic turn is auditable
    assert conv2.state.turns[0].user.startswith("[batch x")
    assert isinstance(metrics.totals, dict)
