"""
Basic exceptions for Gemini batch processing
"""


class GeminiBatchError(Exception):
    """Base exception for Gemini batch processing errors"""

    pass


class APIError(GeminiBatchError):
    """Raised when API calls fail"""

    pass
