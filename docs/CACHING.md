# Context Caching Guide

The Gemini Batch Framework supports both implicit and explicit context caching to reduce API costs and improve performance for repeated content analysis.

## Quick Start

```python
from gemini_batch import BatchProcessor

# Caching is automatically enabled for supported models
processor = BatchProcessor()  # Auto-detects model capabilities

# Or explicitly enable caching
from gemini_batch import GeminiClient
client = GeminiClient.from_env(enable_caching=True)
processor = BatchProcessor(client=client)
```

## Environment Configuration

```bash
# Enable caching globally
GEMINI_ENABLE_CACHING=true

# Model selection affects caching capabilities
GEMINI_MODEL=gemini-2.5-flash-preview-05-20  # Supports implicit + explicit
GEMINI_MODEL=gemini-2.0-flash                # Supports explicit only
```

### Model Caching Capabilities

- **Gemini 2.5 Flash/Pro**: Both implicit and explicit caching
- **Gemini 2.0 Flash**: Explicit caching only
- **Gemini 1.5 Pro/Flash**: Explicit caching only
- **Other models**: Check model documentation for caching support

## Cache Performance Monitoring

```python
# Get cache performance metrics
results = processor.process_questions(content, questions)

# Check cache usage
if results["metrics"]["batch"]["cache_enabled"]:
    cache_ratio = results["metrics"]["batch"]["cache_hit_ratio"]
    print(f"Cache hit ratio: {cache_ratio:.1%}")

    cost_efficiency = results["efficiency"]["cost_efficiency_ratio"]
    print(f"Cost efficiency: {cost_efficiency:.1f}x improvement")

# Detailed cache metrics
cache_metrics = processor.get_cache_efficiency_summary()
if cache_metrics:
    print(f"Active caches: {cache_metrics['active_caches']}")
    print(f"Cache hit rate: {cache_metrics['cache_hit_rate']:.1%}")
```

## Cache Management

```python
# Manual cache cleanup
cleaned = processor.cleanup_caches()
print(f"Cleaned {cleaned} expired caches")

# List active caches (advanced)
from gemini_batch import CacheManager
cache_list = client.list_active_caches()
for cache in cache_list:
    print(f"Cache: {cache['cache_name']}, Usage: {cache['usage_count']}")
```

## Best Practices

- **Large Content**: Most benefit for large documents, videos, or images
- **Repeated Analysis**: Highest efficiency with multiple questions on same content
- **Model Selection**: Gemini 2.5 models provide both implicit and explicit caching
- **TTL Selection**: Use longer TTL for content analyzed repeatedly over time

## Troubleshooting

- **Cache not enabled**: Check model supports caching and `enable_caching=True`
- **Low cache hit ratio**: Content may be below minimum token threshold
- **Cache errors**: Framework automatically falls back to non-cached generation
