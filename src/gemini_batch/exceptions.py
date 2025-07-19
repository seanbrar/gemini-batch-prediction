"""Basic exceptions for Gemini batch processing"""


class GeminiBatchError(Exception):
    """Base exception for Gemini batch processing errors"""


class APIError(Exception):
    """Custom exception for API-related errors"""


class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""


class MissingKeyError(GeminiBatchError):
    """Raised when required API key or configuration key is missing"""


class NetworkError(GeminiBatchError):
    """Raised when network issues occur"""


class FileError(GeminiBatchError):
    """Raised when file operations fail"""


class ValidationError(GeminiBatchError):
    """Raised when input validation fails"""


class UnsupportedContentError(GeminiBatchError):
    """Raised when content type is not supported"""
