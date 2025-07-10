# Enabling Library Logging

The `gemini-batch` library includes a detailed internal logging system to provide insight into its operations, which can be invaluable for troubleshooting complex workflows, understanding caching behavior, and diagnosing API issues.

By default, these logs are silent. This guide shows you how to enable and configure them in your own application.

## Why Enable Logging?

When you encounter unexpected behavior - such as an API error, a strange result, or a performance issue - the library's logs can show you exactly what's happening under the hood. You can see which strategies are being used, how API responses are being parsed, and why certain fallbacks are triggered.

## Basic Console Logging

To see high-level information and warnings from the library, you can enable basic logging. This is the simplest way to get visibility into major operations.

```python
import logging
from gemini_batch import BatchProcessor

# Enable INFO level logging
# This will show INFO, WARNING, ERROR, and CRITICAL messages.
logging.basicConfig(level=logging.INFO)

# Now, when you use the library, you'll see its output
processor = BatchProcessor()
processor.process_questions("some content", ["a question"])

# --- Example Console Output ---
# INFO:gemini_batch.batch_processor:Starting question processing for 1 questions.
# INFO:gemini_batch.batch_processor:Attempting batch processing.
# WARNING:gemini_batch.client.cache_manager:Cache hit for content hash '...' -> cache_name '...'
```

> ***Note:** The exact output will vary based on your code and library version.*

## Debugging with Verbose Output

For deep troubleshooting, you can set the logging level to `DEBUG`. This will produce a large volume of detailed output, showing every step of the library's internal processes.

```python
import logging
from gemini_batch import BatchProcessor

# Enable DEBUG level logging for maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

processor = BatchProcessor()
processor.process_questions("some content", ["a question"])

# --- Example Console Output ---
# 2025-07-09 04:13:00,123 - gemini_batch.batch_processor - INFO - Starting...
# 2025-07-09 04:13:00,124 - gemini_batch.batch_processor - DEBUG - Using BatchPromptBuilder.
# 2025-07-09 04:13:00,125 - gemini_batch.client.cache_manager - DEBUG - Cache analysis result...

```

## Advanced: Fine-Grained Control

The library's loggers are hierarchical. This allows you to control the verbosity of different components independently. For example, you can enable `DEBUG` logging for the entire library but silence a particularly "chatty" component.

```python
import logging
from gemini_batch import BatchProcessor

# Set up a general DEBUG level
logging.basicConfig(level=logging.DEBUG)

# Silence a specific component by setting its level higher
logging.getLogger('gemini_batch.client.cache_manager').setLevel(logging.INFO)

# Now you'll get detailed logs from everywhere *except* the cache manager,
# which will only show INFO and higher.
processor = BatchProcessor()
processor.process_questions("some content", ["a question"])
```

You can also do the reverse - keep logging quiet by default but enable `DEBUG` for just one part of the library you want to inspect:

```python
import logging
from gemini_batch import BatchProcessor

# Set the default level to WARNING (quiet)
logging.basicConfig(level=logging.WARNING)

# Enable detailed logging for just the response processor
logging.getLogger('gemini_batch.response.processor').setLevel(logging.DEBUG)

processor = BatchProcessor()
processor.process_questions("some content", ["a question"])
```

## Logging to a File

To save logs to a file instead of printing them to the console, use the `filename` argument in `basicConfig`.

```python
import logging
from gemini_batch import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gemini_batch.log',
    filemode='w'  # 'w' to overwrite the file each time, 'a' to append
)

logging.info("Starting a new session.")
processor = BatchProcessor()
processor.process_questions("some content", ["a question"])
logging.info("Session finished.")
```

## More Info

The `gemini-batch` library uses Python's standard `logging` module. For more advanced configuration (e.g., rotating log files, sending logs to different handlers, or using dictionary-based configuration), the official documentation is an excellent resource.

- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html) - A great starting point
- [Python Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html) - More advanced recipes and examples
