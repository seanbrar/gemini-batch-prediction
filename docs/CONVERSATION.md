# Conversation Sessions Guide

The conversation system enables contextual multi-turn interactions with persistent memory across questions and sources.

## Quick Start

```python
from gemini_batch import create_conversation

# Create session with single source
session = create_conversation("document.pdf")

# Ask questions with automatic context retention
answer1 = session.ask("What is the main topic?")
answer2 = session.ask("What evidence supports this?")  # References previous answer
```

## Multi-Source Conversations

```python
# Analyze multiple sources together
sources = [
    "research_paper.pdf",
    "https://youtube.com/watch?v=example",
    "data_directory/"
]

session = create_conversation(sources)

# Cross-source analysis with context
session.ask("What are the common themes across all sources?")
session.ask("Which source provides the strongest evidence?")  # References themes
```

## Batch Questions with Context

```python
# Process multiple questions while building conversation history
questions = [
    "What are the main findings?",
    "What methodology was used?", 
    "What are the limitations?"
]

answers = session.ask_multiple(questions)

# Follow-up leverages all previous context
followup = session.ask("Which limitation is most critical?")
```

## Source Management

```python
session = create_conversation("initial_document.pdf")

# Dynamic source management during conversation
session.add_source("supplementary_data.csv")
session.add_source("https://example.com/article")

# Remove sources no longer needed
session.remove_source("initial_document.pdf")

# Check current sources
current_sources = session.list_sources()
```

## Session Persistence

Save and load conversation sessions with full context:

```python
# Save current session
session_id = session.save()  # Auto-generates filename
# Or specify path
session_id = session.save("my_analysis_session.json")

# Load previous session with full history
loaded_session = load_conversation(session_id)

# Continue where you left off
loaded_session.ask("Based on our previous discussion, what's next?")
```

## Session Analytics

```python
# Get conversation statistics
stats = session.get_stats()
print(f"Questions asked: {stats['total_turns']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Sources analyzed: {stats['active_sources']}")
print(f"Session duration: {stats['session_duration']:.1f}s")

# Review conversation history
history = session.get_history()  # List of (question, answer) pairs
detailed = session.get_detailed_history()  # Full metadata
```

## Advanced Configuration

### Custom History Length

Controls how much previous context is included:

```python
# Limit context to last 3 turns (default: 5)
session = create_conversation(sources, max_history_turns=3)
```

### Custom Processor Configuration

```python
from gemini_batch import BatchProcessor

# Use custom processor with specific settings
processor = BatchProcessor(
    model_name="gemini-2.0-flash",
    enable_caching=True
)

session = create_conversation(sources, processor=processor)
```

### System Instructions with Context

```python
# System instruction combines with conversation history
answer = session.ask(
    "Explain this concept",
    system_instruction="You are a physics professor"
)
```

## Error Handling

The system tracks failed interactions while preserving successful context:

```python
try:
    answer = session.ask("Complex question")
except Exception as e:
    # Error is logged, but conversation continues
    stats = session.get_stats()
    print(f"Error rate: {1 - stats['success_rate']:.1%}")
    
# Subsequent questions still work with previous successful context
session.ask("Different question")
```

## Common Patterns

### Research Analysis

```python
# Start with broad analysis
session = create_conversation("research_papers/")
themes = session.ask("What are the main research themes?")

# Drill down with context
methods = session.ask("Which methodologies support these themes?")
future = session.ask("What research directions do these suggest?")
```

### Learning Sessions

```python
# Build understanding progressively
session = create_conversation("textbook.pdf")
session.ask("What are the basic concepts?")
session.ask("How do these concepts relate?")  # Uses previous concepts
session.ask("What's a practical application?")  # Uses relationships
```

### Document Synthesis

```python
# Combine multiple document perspectives
sources = ["report1.pdf", "report2.pdf", "report3.pdf"]
session = create_conversation(sources)

session.ask("What does each report conclude?")
session.ask("Where do they agree or disagree?")  # References conclusions
session.ask("What's the overall consensus?")  # Synthesizes agreement/disagreement
```

## Session Management

```python
# Clear history while keeping sources
session.clear_history()

# Review active sources
sources = session.list_sources()

# Get session identifier
session_id = session.session_id
```
