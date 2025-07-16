# Conversation System Guide

The Gemini Batch Framework supports **conversation sessions** - persistent, stateful interactions that maintain context across multiple questions and sources.

## Quick Start

```python
from gemini_batch import create_conversation

# Create session with sources
session = create_conversation("research_papers/")

# Ask questions with context retention
answers = session.ask_multiple([
    "What are the main research themes?",
    "Which papers discuss efficiency techniques?"
])

# Natural follow-up builds on previous context
followup = session.ask("How do these techniques compare in practice?")
```

## Core Functionality

### Session Management

```python
# Create with single or multiple sources
session = create_conversation("document.pdf")
session = create_conversation(["paper1.pdf", "paper2.pdf", "video.mp4"])

# Ask single or multiple questions
answer = session.ask("What is this about?")
answers = session.ask_multiple(["Q1", "Q2", "Q3"])

# Session maintains context automatically
history = session.get_history()
stats = session.get_stats()
```

### Context Persistence

```python
# Save/load conversation state
session.save("conversation.json")
session = load_conversation("conversation.json")

# Add/remove sources dynamically
session.add_source("new_paper.pdf")
session.remove_source("old_paper.pdf")

# Clear history while keeping sources
session.clear_history()
```

### Advanced Configuration

```python
# Custom processor settings
processor = BatchProcessor(model="gemini-2.0-flash", enable_caching=True)
session = ConversationSession("sources/", _processor=processor, max_history_turns=15)

# System instructions per question
answer = session.ask("Summarize this", system_instruction="You are a research assistant.")
```

## Context Management

### Automatic Token Budgeting

The framework automatically manages context length:

```python
# Framework handles large documents + long history
session = create_conversation("large_document.pdf")

# Automatically:
# - Estimates token usage
# - Truncates history when needed
# - Preserves most recent/relevant context
# - Logs context management decisions

for i in range(20):
    session.ask(f"Question {i}")  # History automatically managed
```

### Context Overflow Handling

```python
# Graceful handling of context overflow
session = create_conversation("very_large_document.pdf")

# Framework:
# - Preserves essential context
# - Logs overflow warnings
# - Continues processing
# - Maintains conversation coherence
```

## Session Data Structure

```json
{
  "session_id": "uuid",
  "sources": ["document.pdf", "https://..."],
  "history": [
    {
      "question": "What is this about?",
      "answer": "This document discusses...",
      "timestamp": "2024-01-01T12:00:00Z",
      "sources_snapshot": ["document.pdf"],
      "cache_info": {"cache_hit_ratio": 0.8}
    }
  ],
  "created_at": "2024-01-01T12:00:00Z"
}
```

## Best Practices

### Efficient Conversations

```python
# Batch related questions for efficiency
session.ask_multiple([
    "What are the main themes?",
    "What are the key findings?",
    "What are the implications?"
])

# Use follow-ups for clarification
session.ask("Can you elaborate on the third point?")

# Save progress for long conversations
if len(session.get_history()) > 10:
    session.save("checkpoint.json")
```

### Source Organization

```python
# Group related sources
session = create_conversation([
    "papers/neural_networks/",
    "papers/attention_mechanisms/",
    "videos/lectures/"
])

# Add sources as conversation progresses
session.add_source("new_paper.pdf")
session.ask("How does this new paper relate to our discussion?")
```

## Troubleshooting

### Common Issues

#### "Context too long" warnings

- Framework automatically truncates history
- Consider saving/loading sessions for very long conversations
- Use `max_history_turns` to limit history size

#### Session loading errors

- Check file permissions and JSON format
- Verify sources still exist
- Consider recreating session with same sources

#### Memory usage

- Large history can consume memory
- Use `clear_history()` periodically
- Consider session checkpoints for very long conversations
