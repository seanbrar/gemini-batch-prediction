# Development Standards

> Project-specific conventions for the Gemini Batch Framework GSoC project

## Architecture Patterns

### Pipeline Data Flow

- **Immutable Commands**: All data structures use `@dataclass(frozen=True)`
- **Handler Pattern**: Handlers implement `BaseAsyncHandler[T_In, T_Out, E]` and return `Result[T_Out, E]` asynchronously
- **Explicit Plans**: Complex logic creates explicit data (`ExecutionPlan`) for simple executors
- **Unidirectional Flow**: Data flows through pipeline stages without backtracking

### Configuration Philosophy

```python
# Explicit (preferred for robust applications/tests)
executor = GeminiExecutor(config=explicit_config)

# Resolved from environment (convenient for scripts/notebooks)
executor = create_executor()  # Uses `resolve_config()` under the hood
```

### Error Handling Conventions

- **Enriched Exceptions**: Always include context in API errors
- **Result Types**: Consider `Success | Failure` for pipeline stages
- **Graceful Degradation**: Plan fallback strategies explicitly in `ExecutionPlan`

## File Organization

### Section Separators

Use this exact format across all configuration and source files:

```python
# --- Section Description ---
```

### Module Responsibilities

- `config/` - Resolution engine and immutable `FrozenConfig`
- `core/` - Data types and fundamental abstractions
- `pipeline/` - Stateless transformation handlers and planners
- `executor.py` - Orchestrates pipeline stages
- `telemetry.py` - Lightweight telemetry context and reporters
- `extensions/` - Optional features (conversation, visualization)

## Project-Specific Naming

### Handler Conventions

```python
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.core.types import Result, InitialCommand, ResolvedCommand
from gemini_batch.core.exceptions import SourceError

class SourceHandler(BaseAsyncHandler[InitialCommand, ResolvedCommand, SourceError]):  # End with "Handler"
    async def handle(self, command: InitialCommand) -> Result[ResolvedCommand, SourceError]:  # Always async
        ...
```

### Factory Functions

```python
def create_executor() -> GeminiExecutor  # Main entrypoint

# Conversation: use the immutable facade
from gemini_batch import create_executor
from gemini_batch.extensions.conversation import Conversation

executor = create_executor()
conversation = Conversation.start(executor, sources=[...])
```

### Telemetry Integration

```python
from gemini_batch.telemetry import TelemetryContext

ctx = TelemetryContext()
with ctx("pipeline.stage_name", **metadata):  # Dot notation for hierarchical scopes
    ...
```

## Documentation Integration

Rather than duplicate existing documentation, this section coordinates:

- **Telemetry**: See Explanation → Concepts (Telemetry), Deep Dives (Telemetry Spec), and ADR-0006 for implementation details and design rationale.
- **Logging**: See `docs/LOGGING.md` for usage patterns
- **Testing**: See `tests/characterization/` for golden master patterns
- **Cache Management**: See `dev/design/cache_manager_planner_pattern.md`

## Code Quality Tools

Configured in `pyproject.toml`:

- **Formatting**: `ruff format` (Google docstring convention)
- **Type Checking**: `mypy` in strict mode
- **Linting**: Comprehensive `ruff` rule set
- **Testing**: `pytest` with golden master characterization tests

## GSoC-Specific Considerations

- **Mentor Reviews**: Focus on clear architectural boundaries and explicit data flow
- **Testability**: Every component should be testable in isolation
- **Documentation**: Architecture decisions should be discoverable and well-reasoned
