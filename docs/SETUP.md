# Gemini Batch API Setup Guide

Quick setup guide for the `gemini_batch` package - initialization methods and basic usage.

## Installation & Setup

```bash
pip install -e .
```

### API Key Configuration

Get your API key from [Google AI Studio](https://ai.dev/) and configure it using one of these methods:

**Environment Variables (Recommended)**
```bash
# Create .env file or export directly
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash  # Optional
GEMINI_TIER=free               # Options: free, tier_1, tier_2, tier_3
```

> **Note**: Free tier limits vary by model (typically 10-30 requests/minute). For higher limits, enable billing in Google AI Studio.

## Initialization Methods

The package provides two main classes: `GeminiClient` for direct API access and `BatchProcessor` for processing multiple questions efficiently.

### 1. Environment-Based (Simplest)

Automatically reads from environment variables:

```python
from gemini_batch import GeminiClient, BatchProcessor

# Uses GEMINI_API_KEY from environment
client = GeminiClient()
processor = BatchProcessor()
```

### 2. Direct Parameters

Explicitly provide configuration:

```python
# Direct initialization
client = GeminiClient(
    api_key="your_api_key_here",
    model_name="gemini-2.0-flash",
    enable_caching=False,        # Optional: enable response caching
    tier=APITier.FREE           # Optional: explicit tier setting
)

processor = BatchProcessor(api_key="your_api_key_here")
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

client = GeminiClient.from_config(config)
```

### 4. Factory Methods

Convenient shortcuts for common patterns:

```python
# Pure environment configuration
client = GeminiClient.from_env()

# Environment with overrides
client = GeminiClient.from_env(model_name="gemini-1.5-pro")
```

### 5. Shared Client

Use one client configuration across multiple components:

```python
client = GeminiClient(api_key="your_key")
processor = BatchProcessor(client=client)  # Reuses client config
```

## Basic Usage

### Simple Content Generation

```python
from gemini_batch import GeminiClient

client = GeminiClient()
response = client.generate_content("What is artificial intelligence?")
print(response)
```

### With System Instructions

```python
response = client.generate_content(
    prompt="Explain quantum physics",
    system_instruction="You are a helpful physics teacher."
)
```

### Getting Usage Metrics

```python
result = client.generate_content("Hello world", return_usage=True)
print(f"Tokens used: {result['usage']['total_tokens']}")
print(f"Response: {result['text']}")
```

### Basic Batch Processing

```python
from gemini_batch import BatchProcessor

processor = BatchProcessor()
content = "Artificial intelligence is transforming technology..."
questions = ["What is AI?", "How does it work?"]

results = processor.process_questions(content, questions)
for i, answer in enumerate(results['answers'], 1):
    print(f"Q{i}: {answer}")
```

### Multiple Content Types

The `process_questions()` method supports text, files, URLs, and mixed content. See [`SOURCE_HANDLING.md`](SOURCE_HANDLING.md) for detailed examples.

## Configuration Validation

Check your setup before processing:

```python
client = GeminiClient()
config = client.get_config_summary()
print(f"Using {config['tier_name']} with {config['client_model_name']}")
print(f"API key configured: {config['api_key_present']}")
```

## Error Handling

```python
from gemini_batch.exceptions import MissingKeyError, APIError

try:
    client = GeminiClient()
    response = client.generate_content("Hello")
except MissingKeyError as e:
    print(f"API key required: {e}")
except APIError as e:
    print(f"API call failed: {e}")
```

## Troubleshooting

### Common Issues

**"API key required" error**
```python
# Verify your API key is set
import os
print("API key present:", bool(os.getenv('GEMINI_API_KEY')))
```

**Rate limit errors**
```python
# Check your current tier configuration
client = GeminiClient()
config = client.get_config_summary()
print(f"Tier: {config['tier_name']}")
print(f"Rate limit: {client.rate_limit_requests} requests/minute")
```

**Model not available errors**
- Some models require billing enabled (TIER_1+)
- Use `config.get_model_limits(model_name)` to check availability

---

This covers the essential setup and initialization patterns. For advanced batch processing features and efficiency analysis, see the examples in the `examples/` directory.