# Developer Guide: Logging in the gemini-batch Library

**Audience:** Internal Developers, GSoC Contributors
**Purpose:** To document the design, strategy, and philosophy of the logging system. This guide ensures that future contributions adhere to a consistent, robust, and maintainable logging architecture.

## 1. Core Philosophy: A Library Must Be a Good Citizen

The single most important principle governing our logging system is this: **A library should never configure logging handlers for the user.**

This means our library code must **never** call `logging.basicConfig()`, nor should it add handlers like `logging.StreamHandler` or `logging.FileHandler` to any logger.

**Why is this the Golden Rule?**

The application developer who uses our library is the sole owner of their application's logging strategy. They decide what log messages are important, what format they should be in, and where they should go (to the console, a file, a remote service, etc.).

If our library were to configure logging, it could inadvertently:

- Hijack or overwrite the application's existing logging setup.
- Create duplicate log messages.
- Send logs to an unexpected location (e.g., a file the user doesn't want).

This "good citizen" approach is a standard best practice in the Python ecosystem, followed by major libraries like `requests`, `urllib3`, and `SQLAlchemy`. Our goal is to provide rich, detailed logging information without imposing any configuration on the user.

## 2. Architectural Design

Our logging architecture is designed for simplicity, control, and maintainability. It is built on two key components.

### 2.1. The `NullHandler` Guard

To prevent the annoying `No handlers could be found for logger "..."` message from appearing when a user has no logging configured, we attach a `logging.NullHandler` to our library's top-level logger.

This is done once in the library's root `__init__.py` file:

**File:** `gemini_batch/__init__.py`

```python
import logging

# Set up a null handler for the library's root logger.
# This prevents 'No handler found' errors if the consuming app has no logging configured.
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

The `NullHandler` acts as a "do-nothing" safety net. It receives all log events from our library and silently discards them. If the user does configure their own handlers, those handlers will process the events as expected, and the `NullHandler` will be effectively ignored.

### 2.2. Namespaced, Hierarchical Loggers

The core of our design is the use of namespaced loggers that mirror the project's module structure. This is achieved by instantiating a logger in each module using a consistent pattern:

```python
# At the top of any module, e.g., gemini_batch/client/cache_manager.py
import logging
log = logging.getLogger(__name__)
```

The `__name__` variable automatically resolves to the module's dotted path (e.g., `gemini_batch.client.cache_manager`). This creates a powerful logger hierarchy. A developer using the library can then exercise fine-grained control over the log output.

For example, a user can:

- **Silence the entire library:**

    ```python
    logging.getLogger('gemini_batch').setLevel(logging.WARNING)
    ```

- **Enable verbose debugging for a specific, complex component:**

    ```python
    logging.getLogger('gemini_batch.client.cache_manager').setLevel(logging.DEBUG)
    ```

- **See high-level info from the batch processor but nothing else:**

    ```python
    logging.getLogger('gemini_batch.batch_processor').setLevel(logging.INFO)
    ```

This hierarchical approach is fundamental to providing debugging insight without creating overwhelming noise.

## 3. Logging Strategy and Best Practices

To ensure our logs are useful and consistent, we adhere to the following strategic guidelines.

### 3.1. Choosing the Right Log Level

The log level communicates the severity and intent of a message. Using levels consistently is key to making the logs filterable and easy to understand.

- `DEBUG`
  - **Purpose:** Detailed, high-volume information useful only for deep-diving into the library's internal state.
  - **Examples:**
    - `log.debug("Cache hit for content hash '%s'", content_hash[:8])`
    - `log.debug("Determined processing strategy '%s' for source '%s'", strategy, path)`
    - `log.debug("Extracting answers from response. type=%s, preview='%s...'", ...)`

- `INFO`
  - **Purpose:** High-level confirmation of significant operations, state changes, or milestones. These logs should be sparse enough to be readable in a production environment.
  - **Examples:**
    - `log.info("Starting question processing for %d questions.", len(questions))`
    - `log.info("Creating new explicit cache for model '%s' with TTL %ds.", model, ttl)`
    - `log.info("Batch processing completed successfully. Token efficiency: %.2fx", ratio)`

- `WARNING`
  - **Purpose:** An unexpected event occurred, or a recoverable error was handled. The operation may have fallen back to a less efficient path, but the library is still functioning correctly.
  - **Examples:**
    - `log.warning("Batch processing failed, falling back to individual calls. Reason: %s", e)`
    - `log.warning("Failed to decode JSON from response. Treating as single raw text answer.")`
    - `log.warning("API call failed with retryable error. Retrying in %.2fs", delay)`

- `ERROR`
  - **Purpose:** A serious problem occurred that prevented a specific operation from completing successfully.
  - **Examples:**
    - `log.error("Response JSON did not match the provided schema.", exc_info=True)`
    - `log.error("API call failed with non-retryable error.", exc_info=True)`

- `CRITICAL`
  - **Purpose:** A severe error that threatens to terminate the entire application. **This level should almost never be used in a library.**

### 3.2. Logging Exceptions Correctly

When catching an exception, it is crucial to log the full stack trace. This provides the complete context needed to debug the error.

- **Best Practice:** Use `log.exception()` or `log.error(..., exc_info=True)`.

```python
# Good: Captures the full traceback
try:
    # ... some operation that might fail
except Exception as e:
    log.exception("An unexpected error occurred during response parsing.")

# Also good:
except ValidationError as e:
    log.error("Response JSON did not match the provided schema.", exc_info=True)
```

This is far more useful than simply logging the exception message, e.g., `log.error(f"Error: {e}")`, which loses all context of where the error occurred.

### 3.3. Writing Effective Log Messages

- **Be Concise but Informative:** Include key variables and context.
- **Use Modern Formatting:** Use `%` style formatting or f-strings. The `%` style is slightly preferred as it defers string formatting until the message is actually processed by a handler, which can be a minor performance benefit.
- **Do Not Log Sensitive Data:** Never log raw API keys, user credentials, or full, verbose content that might contain private information.

## 4. How to Use This System for Debugging

As a developer working on this library, you can easily enable and configure the logs for your testing and troubleshooting needs.

Create a simple script (e.g., `run_debug.py`) outside the library source tree:

```python
# run_debug.py
import logging
from gemini_batch import BatchProcessor

# 1. Configure logging for your debugging session.
# This setup is for YOUR TEST APPLICATION, not the library itself.
logging.basicConfig(
    # Set the level to DEBUG to see everything.
    level=logging.DEBUG,
    # A detailed format is great for debugging.
    format='%(asctime)s - %(name)-40s - %(levelname)-8s - %(message)s',
    # Optional: Log to a file instead of the console
    # filename='debug.log',
    # filemode='w'
)

# 2. (Optional) To reduce noise, you can silence overly verbose modules.
# For example, if the cache manager is too chatty for your current task:
# logging.getLogger('gemini_batch.client.cache_manager').setLevel(logging.INFO)

# 3. Now, run your library's code
def main():
    """Run a test scenario."""
    print("\n--- Running Library Test ---")
    try:
        # Your library's code will now emit detailed logs to the console (or file)
        processor = BatchProcessor()
        processor.process_questions("some_content.txt", ["What is this?"])
    except Exception:
        # Catch exceptions and log them with full context
        logging.exception("The test run failed with an unhandled exception.")

if __name__ == "__main__":
    main()
```

By running this script, you gain complete, filterable insight into the library's internal operations, making it significantly easier to trace complex logic and pinpoint the source of bugs.
