"""
Project-wide constants for Gemini Batch Processing Framework

This module centralizes all magic numbers and configuration constants
used throughout the project for consistency and maintainability.
"""

# API and Network Constants
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
RATE_LIMIT_RETRY_BASE_DELAY = 5.0  # seconds
NETWORK_TIMEOUT = 30.0  # seconds
RATE_LIMIT_WINDOW = 60  # seconds

# Conservative fallback rate limits (when config detection fails)
FALLBACK_REQUESTS_PER_MINUTE = 15
FALLBACK_TOKENS_PER_MINUTE = 250_000

# File Processing Constants
DEFAULT_FILE_LIFETIME = 3600  # 1 hour in seconds
DEFAULT_FILE_PROCESSING_TIMEOUT = 120.0  # 2 minutes
DEFAULT_POLLING_INTERVAL = 2.0  # seconds
DEFAULT_SCANNER_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
PDF_PAGE_WARNING_THRESHOLD = 500  # warn for large documents

# File Size Limits
DEFAULT_MAX_SIZE = 100 * 1024 * 1024  # 100MB
TEXT_MAX_SIZE = 50 * 1024 * 1024  # 50MB
GEMINI_PDF_MAX_SIZE = 20 * 1024 * 1024  # 20MB
GEMINI_PDF_MAX_PAGES = 1000
GEMINI_FILES_API_MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

# Batch Processing Constants
TARGET_EFFICIENCY_RATIO = 3.0  # Minimum efficiency improvement target
MIN_QUALITY_SCORE = 0.8  # Minimum acceptable quality score

# Visualization Constants
VIZ_FIGURE_SIZE = (15, 10)
VIZ_SCALING_FIGURE_SIZE = (15, 6)
VIZ_ALPHA = 0.8
VIZ_BAR_WIDTH = 0.35

# Visualization Colors
VIZ_COLORS = {
    "individual": "#ff7f7f",
    "batch": "#7fbf7f",
    "improvements": ["#4CAF50", "#2196F3", "#FF9800"],
    "line": "#2E7D32",
}
