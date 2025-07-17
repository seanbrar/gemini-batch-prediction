"""
Basic exceptions for Gemini batch processing
"""


class GeminiBatchError(Exception):
    """Base exception for Gemini batch processing errors"""

    pass


class APIError(Exception):
    """Custom exception for API-related errors"""

    pass


class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""

    pass


class MissingKeyError(GeminiBatchError):
    """Raised when required API key or configuration key is missing"""

    pass


class NetworkError(GeminiBatchError):
    """Raised when network issues occur"""

    pass


class FileError(GeminiBatchError):
    """Raised when file operations fail"""

    pass


class ValidationError(GeminiBatchError):
    """Raised when input validation fails"""

    pass


class UnsupportedContentError(GeminiBatchError):
    """Raised when content type is not supported"""

    pass
