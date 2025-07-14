# Gemini Batch API Setup Guide

Quick setup guide for the `gemini_batch` package - initialization methods and basic usage.

## Installation & Setup

```bash
pip install -e .
```

### API Key Configuration

Get your API key from [Google AI Studio](https://ai.dev/) and configure it using one of these methods:

#### Environment Variables (Recommended)

```bash
# Create .env file or export directly
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash       # Optional
GEMINI_TIER=free                    # Options: free, tier_1, tier_2, tier_3
GEMINI_ENABLE_CACHING=true          # Enable context caching for cost optimization
```

> **Note**: Free tier limits vary by model (typically 10-30 requests/minute). For higher limits, enable billing in Google AI Studio.

## Initialization Methods

The package provides two main classes: `BatchProcessor` for most use cases and `GeminiClient` for advanced scenarios.

### 1. Environment-Based (Simplest)

Automatically reads from environment variables:

```python
from gemini_batch import BatchProcessor

# Uses GEMINI_API_KEY from environment
processor = BatchProcessor()
```

### 2. Direct Parameters

Explicitly provide configuration:

```python
# Direct initialization with parameters
processor = BatchProcessor(
    api_key="your_api_key_here",
    model="gemini-2.0-flash",
    enable_caching=False,        # Optional: enable response caching
    tier="free"                  # Optional: explicit tier setting
)

# Note: The 'tier' parameter accepts either a string ("free", "tier_1", etc.) or the APITier enum (APITier.FREE, APITier.TIER_1, ...).
```

### 3. ConfigManager (Advanced)

For fine-grained control over API tiers and rate limits:

```python
from gemini_batch.config import ConfigManager, APITier

config = ConfigManager(
    api_key="your_api_key_here",
    model="gemini-2.0-flash",
    tier=APITier.FREE
)

# Pass config parameters to processor
processor = BatchProcessor(
    api_key=config.api_key,
    model=config.model,
    tier=config.tier,
    enable_caching=config.enable_caching
)
```

### 4. Advanced: Direct Client Usage

For advanced scenarios requiring direct API access:

```python
from gemini_batch import GeminiClient, BatchProcessor

# Create client for advanced use cases
client = GeminiClient(api_key="your_key")
processor = BatchProcessor(_client=client)  # Reuses client config
```

## Basic Usage

### Batch Processing (Recommended)

```python
from gemini_batch import BatchProcessor

processor = BatchProcessor()
content = "Artificial intelligence is transforming technology..."
questions = ["What is AI?", "How does it work?"]

results = processor.process_questions(content, questions)
for i, answer in enumerate(results['answers'], 1):
    print(f"Q{i}: {answer}")
```

### With System Instructions

```python
results = processor.process_questions(
    content="Explain quantum physics",
    questions=["What are the key principles?"],
    system_instruction="You are a helpful physics teacher."
)
```

### Getting Usage Metrics

```python
results = processor.process_questions(
    content="Hello world", 
    questions=["What is this?"], 
    return_usage=True
)
print(f"Tokens used: {results['metrics']['batch']['total_tokens']}")
```

### Multiple Content Types

The `process_questions()` method supports text, files, URLs, and mixed content. See [`SOURCE_HANDLING.md`](SOURCE_HANDLING.md) for detailed examples.

## Configuration Validation

Check your setup before processing:

```python
processor = BatchProcessor()
config = processor.client.get_config_summary()
print(f"Using {config['tier_name']} with {config['model']}")
print(f"API key configured: {config['api_key_present']}")
```

## Error Handling

```python
from gemini_batch.exceptions import APIError

try:
    processor = BatchProcessor()
    results = processor.process_questions("Hello", ["What is this?"])
except ValueError as e:
    print(f"Configuration error: {e}")
except APIError as e:
    print(f"API call failed: {e}")
```

## Troubleshooting

### Common Issues

#### "API key required" error

```python
# Verify your API key is set
import os
print("API key present:", bool(os.getenv('GEMINI_API_KEY')))
```

#### Rate limit errors

```python
# Check your current tier configuration
processor = BatchProcessor()
config = processor.client.get_config_summary()
print(f"Tier: {config['tier_name']}")
print(f"Rate limit: {processor.client.rate_limit_requests} requests/minute")
```

#### Model not available errors

- Some models require billing enabled (TIER_1+)
- Use `config.get_model_limits(model_name)` to check availability

---

This covers the essential setup and initialization patterns. For advanced batch processing features and efficiency analysis, see the examples in the `examples/` directory.
