"""Basic exceptions for Gemini batch processing"""  # noqa: D415


class GeminiBatchError(Exception):
    """Base exception for Gemini batch processing errors"""  # noqa: D415


class APIError(Exception):
    """Custom exception for API-related errors"""  # noqa: D415


class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""  # noqa: D415


class MissingKeyError(GeminiBatchError):
    """Raised when required API key or configuration key is missing"""  # noqa: D415


class NetworkError(GeminiBatchError):
    """Raised when network issues occur"""  # noqa: D415


class FileError(GeminiBatchError):
    """Raised when file operations fail"""  # noqa: D415


class ValidationError(GeminiBatchError):
    """Raised when input validation fails"""  # noqa: D415


class UnsupportedContentError(GeminiBatchError):
    """Raised when content type is not supported"""  # noqa: D415
