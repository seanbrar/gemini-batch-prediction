# Enabling Library Logging

Enable detailed internal logging for troubleshooting API issues, understanding caching behavior, and diagnosing unexpected results.

## Quick Start

```python
import logging
from gemini_batch import BatchProcessor

# Basic logging - shows major operations
logging.basicConfig(level=logging.INFO)

processor = BatchProcessor()
processor.process_questions("content", ["question"])

# Output: INFO:gemini_batch.batch_processor:Starting question processing...
```

## Debug Mode

For deep troubleshooting, use DEBUG level to see every internal operation:

```python
import logging
from gemini_batch import BatchProcessor

# Verbose logging - shows all internal operations
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

processor = BatchProcessor()
processor.process_questions("content", ["question"])
```

## Component-Specific Logging

The library uses hierarchical loggers, allowing independent control of different components:

```python
import logging
from gemini_batch import BatchProcessor

# Enable DEBUG for entire library
logging.basicConfig(level=logging.DEBUG)

# Silence chatty components
logging.getLogger('gemini_batch.client.cache_manager').setLevel(logging.INFO)

processor = BatchProcessor()
processor.process_questions("content", ["question"])
```

```python
# Alternative: Debug only specific components
logging.basicConfig(level=logging.WARNING)  # Quiet by default
logging.getLogger('gemini_batch.response.processor').setLevel(logging.DEBUG)

processor = BatchProcessor()
processor.process_questions("content", ["question"])
```

## File Logging

```python
import logging
from gemini_batch import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gemini_batch.log',
    filemode='w'  # 'w' to overwrite, 'a' to append
)

logging.info("Starting session")
processor = BatchProcessor()
processor.process_questions("content", ["question"])
logging.info("Session finished")
```

## Common Use Cases

### Cache Debugging

Shows cache hits/misses and content hashing decisions:

```python
logging.getLogger('gemini_batch.client.cache_manager').setLevel(logging.DEBUG)
```

### API Response Analysis

Reveals response parsing and error handling:

```python
logging.getLogger('gemini_batch.response.processor').setLevel(logging.DEBUG)
```

### Batch Processing Flow

Displays question processing strategies and fallback logic:

```python
logging.getLogger('gemini_batch.batch_processor').setLevel(logging.DEBUG)
```

## References

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
