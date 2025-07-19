"""
Project-wide constants for Gemini Batch Processing Framework
"""  # noqa: D200, D212, D415

# ==============================================================================
# API and Network Configuration
# ==============================================================================

# Retry and timeout settings
MAX_RETRIES = 2
RETRY_BASE_DELAY = 1.0  # seconds
RATE_LIMIT_RETRY_DELAY = 5.0  # seconds
NETWORK_TIMEOUT = 30.0  # seconds
RATE_LIMIT_WINDOW = 60  # seconds

# Conservative fallback rate limits (when config detection fails)
FALLBACK_REQUESTS_PER_MINUTE = 15
FALLBACK_TOKENS_PER_MINUTE = 250_000

# ==============================================================================
# File Processing Configuration
# ==============================================================================

# File lifecycle settings
FILE_LIFETIME = 3600  # 1 hour in seconds
FILE_PROCESSING_TIMEOUT = 300  # 5 minutes in seconds
FILE_POLL_INTERVAL = 2.0  # seconds

# File size limits (in bytes)
_MB = 1024 * 1024
_GB = 1024 * _MB

MAX_FILE_SIZE = 100 * _MB  # General file size limit
MAX_TEXT_SIZE = 50 * _MB  # Text file specific limit
MAX_PDF_SIZE = 20 * _MB  # PDF file specific limit
MAX_PDF_PAGES = 1000  # PDF page count limit
MAX_FILES_API_SIZE = 2 * _GB  # Files API upload limit

# Processing thresholds
FILES_API_THRESHOLD = 20 * _MB  # Size threshold for Files API vs inline
SCANNER_MAX_SIZE = MAX_FILE_SIZE  # Directory scanner file size limit
PDF_PAGE_WARNING = 500  # Warn for large PDF documents

# ==============================================================================
# Batch Processing Configuration
# ==============================================================================

TARGET_EFFICIENCY_RATIO = 3.0  # Minimum efficiency improvement target
MIN_QUALITY_SCORE = 0.8  # Minimum acceptable quality score

# ==============================================================================
# Visualization Configuration
# ==============================================================================

# Figure dimensions
VIZ_FIGURE_SIZE = (15, 10)
VIZ_SCALING_FIGURE_SIZE = (15, 6)

# Styling
VIZ_ALPHA = 0.8
VIZ_BAR_WIDTH = 0.35

# Color scheme
VIZ_COLORS = {
    "individual": "#ff7f7f",
    "batch": "#7fbf7f",
    "improvements": ["#4CAF50", "#2196F3", "#FF9800"],
    "line": "#2E7D32",
}

# ==============================================================================
# Context Caching Configuration
# ==============================================================================

# Cache lifecycle settings
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_TTL = 24 * 3600  # 24 hours maximum
MIN_CACHE_TTL = 300  # 5 minutes minimum

# Cache size thresholds
MIN_CACHEABLE_TOKENS = 1024  # Minimum tokens worth caching
MIN_CACHING_THRESHOLD = 4096
CACHING_VALIDATION_THRESHOLD = 32_768
LARGE_CONTENT_THRESHOLD = 50_000  # Tokens considered "large content"
HUGE_CONTENT_THRESHOLD = 100_000  # Tokens considered "huge content"

# Text analysis thresholds for caching decisions
MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS = 2000  # Minimum text length to analyze for caching
LARGE_TEXT_THRESHOLD = 1000  # Text length threshold for separating content vs prompts

# Cache performance settings
CACHE_CLEANUP_INTERVAL = 300  # 5 minutes between cleanup cycles
MAX_ACTIVE_CACHES = 100  # Maximum number of caches to track
CACHE_HIT_RATIO_THRESHOLD = 0.1  # Minimum cache hit ratio to consider successful

# Cache efficiency targets
TARGET_CACHE_EFFICIENCY = 2.0  # Target cache efficiency improvement
CACHE_BENEFIT_THRESHOLD = 0.1  # 10% improvement threshold for recommendations
