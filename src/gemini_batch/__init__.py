"""Core components for the Gemini Batch Pipeline."""

import importlib.metadata
import logging

# Core pipeline architecture imports
from gemini_batch.core.exceptions import (
    ConfigurationError,
    FileError,
    GeminiBatchError,
    PipelineError,
    SourceError,
    UnsupportedContentError,
    ValidationError,
)
from gemini_batch.core.models import (
    APITier,
    CachingCapabilities,
    ModelCapabilities,
    RateLimits,
    can_use_caching,
    get_model_capabilities,
    get_rate_limits,
)
from gemini_batch.core.types import (
    APICall,
    ConversationTurn,
    ExecutionPlan,
    Failure,
    FinalizedCommand,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Result,
    Source,
    Success,
)
from gemini_batch.executor import GeminiExecutor, create_executor
from gemini_batch.extensions.conversation import (
    BasePersistenceHandler,
    ConversationManager,
    JSONPersistenceHandler,
)
from gemini_batch.extensions.visualization import (
    create_efficiency_visualizations,
    create_focused_efficiency_visualization,
    run_efficiency_experiment,
    visualize_scaling_results,
)

# Version handling
try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "development"

# Set up a null handler for the library's root logger.
# This prevents 'No handler found' errors if the consuming app has no logging configured.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
__all__ = [  # noqa: RUF022
    # Core Executor
    "GeminiExecutor",
    "create_executor",
    # Conversation Management
    "ConversationManager",
    "BasePersistenceHandler",
    "JSONPersistenceHandler",
    # Core Types & Data Models
    "ConversationTurn",
    "InitialCommand",
    "ResolvedCommand",
    "PlannedCommand",
    "FinalizedCommand",
    "APICall",
    "ExecutionPlan",
    "Source",
    "Result",
    "Success",
    "Failure",
    # Model Capabilities
    "ModelCapabilities",
    "CachingCapabilities",
    "RateLimits",
    "APITier",
    "get_model_capabilities",
    "get_rate_limits",
    "can_use_caching",
    # Exceptions
    "GeminiBatchError",
    "PipelineError",
    "ConfigurationError",
    "SourceError",
    "FileError",
    "ValidationError",
    "UnsupportedContentError",
    # Visualization Extensions
    "create_efficiency_visualizations",
    "create_focused_efficiency_visualization",
    "run_efficiency_experiment",
    "visualize_scaling_results",
]
