"""
Centralized file type detection, MIME type handling, and validation utilities
"""

from dataclasses import dataclass
from functools import lru_cache
import mimetypes
from pathlib import Path
import sys
from typing import Optional, Set, Tuple, Union

from ..constants import (
    FILES_API_THRESHOLD,
    MAX_FILE_SIZE,
    MAX_FILES_API_SIZE,
    MAX_PDF_SIZE,
    MAX_TEXT_SIZE,
)
from .scanner import FileType

# Initialize mimetypes database
mimetypes.init()


@dataclass(frozen=True)
class FileTypeInfo:
    """Consolidated file type information"""

    mime_types: Set[str]
    extensions: Set[str]
    gemini_native: bool = True  # Whether natively supported by Gemini API
    preferred_mime_map: Optional[dict] = (
        None  # Extension -> preferred MIME type mapping
    )


# Centralized file type definitions with preferred MIME mappings
_FILE_TYPE_INFO = {
    FileType.TEXT: FileTypeInfo(
        mime_types={
            # Gemini-native text formats
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "application/xml",
            "text/xml",
            "text/x-python",
            "text/javascript",
            "text/html",
            "text/css",
            "text/rtf",
            # Other text formats (not officially supported by Gemini)
            "text/x-rst",
            "application/yaml",
            "text/yaml",
            "text/x-yaml",
            "application/x-sql",
            "text/x-log",
        },
        extensions={
            # Gemini-native extensions
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".xml",
            ".py",
            ".js",
            ".html",
            ".css",
            ".rtf",
            # Other text extensions
            ".rst",
            ".log",
            ".yaml",
            ".yml",
            ".sql",
        },
        gemini_native=False,  # Mixed - some are native, some aren't
        preferred_mime_map={
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "text/javascript",
            ".py": "text/x-python",
            ".rtf": "text/rtf",
            ".csv": "text/csv",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".sql": "application/x-sql",
            ".rst": "text/x-rst",
            ".log": "text/x-log",
        },
    ),
    FileType.PDF: FileTypeInfo(
        mime_types={"application/pdf"},
        extensions={".pdf"},
        preferred_mime_map={".pdf": "application/pdf"},
    ),
    FileType.IMAGE: FileTypeInfo(
        mime_types={
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/svg+xml",
            "image/webp",
            "image/x-icon",
        },
        extensions={".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg", ".webp"},
        preferred_mime_map={
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        },
    ),
    FileType.VIDEO: FileTypeInfo(
        mime_types={
            "video/mp4",
            "video/avi",
            "video/x-msvideo",
            "video/quicktime",
            "video/x-ms-wmv",
            "video/x-flv",
            "video/webm",
            "video/x-matroska",
        },
        extensions={".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"},
        preferred_mime_map={
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
            ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".m4v": "video/mp4",
        },
    ),
    FileType.AUDIO: FileTypeInfo(
        mime_types={
            "audio/mpeg",
            "audio/wav",
            "audio/flac",
            "audio/aac",
            "audio/ogg",
            "audio/x-ms-wma",
            "audio/mp4",
        },
        extensions={".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"},
        preferred_mime_map={
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".wma": "audio/x-ms-wma",
            ".m4a": "audio/mp4",
        },
    ),
}

# Generate mappings programmatically
MIME_TYPE_MAPPINGS = {ft: info.mime_types for ft, info in _FILE_TYPE_INFO.items()}
EXTENSION_MAPPINGS = {ft: info.extensions for ft, info in _FILE_TYPE_INFO.items()}

# Reverse mappings
MIME_TO_FILETYPE = {
    mime: ft for ft, mimes in MIME_TYPE_MAPPINGS.items() for mime in mimes
}
EXTENSION_TO_FILETYPE = {
    ext: ft for ft, exts in EXTENSION_MAPPINGS.items() for ext in exts
}


# Lazy evaluation for expensive computations
@lru_cache(maxsize=1)
def _get_extension_to_mime_mapping():
    """Generate extension to MIME type mapping from file type info"""
    mapping = {}
    for info in _FILE_TYPE_INFO.values():
        if info.preferred_mime_map:
            mapping.update(info.preferred_mime_map)
    return mapping


@lru_cache(maxsize=1)
def _get_gemini_native_text_constants():
    """Generate Gemini-native text constants from file type info"""
    # Define Gemini-native subsets
    native_mime_types = {
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
        "text/xml",
        "text/x-python",
        "text/javascript",
        "text/html",
        "text/css",
        "text/rtf",
    }
    native_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".py",
        ".js",
        ".html",
        ".css",
        ".rtf",
    }

    return native_mime_types, native_extensions


@lru_cache(maxsize=1)
def _get_gemini_native_mime_types():
    """Generate all Gemini-native MIME types"""
    native_text_mime_types, _ = _get_gemini_native_text_constants()
    return native_text_mime_types | {
        mime
        for ft in [FileType.PDF, FileType.IMAGE, FileType.VIDEO, FileType.AUDIO]
        for mime in MIME_TYPE_MAPPINGS[ft]
    }


# Initialize module-level constants using lazy evaluation
current_module = sys.modules[__name__]
current_module.EXTENSION_TO_MIME = _get_extension_to_mime_mapping()
current_module.GEMINI_NATIVE_TEXT_MIME_TYPES = _get_gemini_native_text_constants()[0]
current_module.GEMINI_NATIVE_TEXT_EXTENSIONS = _get_gemini_native_text_constants()[1]
current_module.GEMINI_NATIVE_MIME_TYPES = _get_gemini_native_mime_types()

# Consolidated sets
ALL_SUPPORTED_MIME_TYPES = set().union(*MIME_TYPE_MAPPINGS.values())
ALL_SUPPORTED_EXTENSIONS = set().union(*EXTENSION_MAPPINGS.values())
TEXT_MIME_TYPES = MIME_TYPE_MAPPINGS[FileType.TEXT]
OTHER_TEXT_MIME_TYPES = TEXT_MIME_TYPES - current_module.GEMINI_NATIVE_TEXT_MIME_TYPES
OTHER_TEXT_EXTENSIONS = (
    EXTENSION_MAPPINGS[FileType.TEXT] - current_module.GEMINI_NATIVE_TEXT_EXTENSIONS
)


def get_mime_type(file_path: Path, use_magic: bool = True) -> Optional[str]:
    """Get MIME type using content-based detection or extension fallback"""
    if use_magic:
        try:
            import magic

            return magic.Magic(mime=True).from_file(str(file_path))
        except (ImportError, Exception):
            pass

    # Try our preferred mapping first, then fall back to mimetypes
    extension = file_path.suffix.lower()
    if extension in current_module.EXTENSION_TO_MIME:
        return current_module.EXTENSION_TO_MIME[extension]

    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type


def determine_file_type(
    file_path: Path, mime_type: Optional[str] = None
) -> Tuple[FileType, Optional[str]]:
    """Determine file type using MIME type with extension fallback"""
    mime_type = mime_type or get_mime_type(file_path)

    if mime_type:
        base_mime = mime_type.split(";")[0].strip()

        # Direct mapping
        if base_mime in MIME_TO_FILETYPE:
            return MIME_TO_FILETYPE[base_mime], mime_type

        # Generic category mapping
        for prefix, file_type in [
            ("text/", FileType.TEXT),
            ("image/", FileType.IMAGE),
            ("video/", FileType.VIDEO),
            ("audio/", FileType.AUDIO),
        ]:
            if base_mime.startswith(prefix):
                return file_type, mime_type

    # Extension fallback
    extension = file_path.suffix.lower()
    file_type = EXTENSION_TO_FILETYPE.get(extension, FileType.UNKNOWN)
    return file_type, mime_type


def _check_gemini_native(
    file_type: FileType,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
    text_only: bool = False,
) -> bool:
    """Helper for Gemini-native checks"""
    if text_only and file_type != FileType.TEXT:
        return False

    # Non-text files are Gemini-native
    if not text_only and file_type in {
        FileType.PDF,
        FileType.IMAGE,
        FileType.VIDEO,
        FileType.AUDIO,
    }:
        return True

    # Text file checks
    if file_type == FileType.TEXT:
        if mime_type:
            base_mime = mime_type.split(";")[0].strip()
            return base_mime in current_module.GEMINI_NATIVE_TEXT_MIME_TYPES
        if extension:
            return extension.lower() in current_module.GEMINI_NATIVE_TEXT_EXTENSIONS

    # MIME type fallback
    if mime_type:
        base_mime = mime_type.split(";")[0].strip()
        return base_mime in current_module.GEMINI_NATIVE_MIME_TYPES

    return False


def is_gemini_native_text(
    file_type: FileType,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
) -> bool:
    """Check if a text file is natively supported by Gemini API"""
    return _check_gemini_native(file_type, mime_type, extension, text_only=True)


def is_gemini_native_file(
    file_type: FileType,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
) -> bool:
    """Check if file is natively supported by Gemini API"""
    return _check_gemini_native(file_type, mime_type, extension, text_only=False)


def is_text_file(file_type: FileType, mime_type: Optional[str] = None) -> bool:
    """Check if file is text-based"""
    if file_type == FileType.TEXT:
        return True
    if mime_type:
        base_mime = mime_type.split(";")[0].strip()
        return base_mime in TEXT_MIME_TYPES or base_mime.startswith("text/")
    return False


def requires_files_api(file_size: int) -> bool:
    """Check if file size requires Files API (>20MB) vs inline submission"""
    return file_size > FILES_API_THRESHOLD


def is_supported_file(
    file_path: Optional[Path] = None,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
    file_type: Optional[FileType] = None,
) -> bool:
    """Check if a file is supported by the framework"""
    # Determine properties from file_path if needed
    if file_path:
        mime_type = mime_type or get_mime_type(file_path)
        extension = extension or file_path.suffix.lower()
        file_type = file_type or determine_file_type(file_path, mime_type)[0]

    # Check file type
    if file_type and file_type != FileType.UNKNOWN:
        return True

    # Check MIME type
    if mime_type:
        base_mime = mime_type.split(";")[0].strip()
        if base_mime in ALL_SUPPORTED_MIME_TYPES:
            return True

    # Check extension
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        if extension.lower() in ALL_SUPPORTED_EXTENSIONS:
            return True

    return False


def validate_file_size(
    file_path: Path, file_type: FileType, for_gemini_api: bool = False
) -> Tuple[bool, Optional[str]]:
    """Validate file size against appropriate limits"""
    try:
        size = file_path.stat().st_size
    except OSError as e:
        return False, f"Cannot access file: {e}"

    # Size limit checks with early returns
    limits = [
        (
            for_gemini_api and size > MAX_FILES_API_SIZE,
            f"File too large for Gemini API: {size / (1024**3):.1f}GB (max: 2GB)",
        ),
        (
            for_gemini_api and file_type == FileType.PDF and size > MAX_PDF_SIZE,
            f"PDF too large: {size / (1024**2):.1f}MB (max: 20MB)",
        ),
        (
            file_type == FileType.TEXT and size > MAX_TEXT_SIZE,
            f"Text file too large: {size / (1024**2):.1f}MB (max: 50MB)",
        ),
        (
            size > MAX_FILE_SIZE,
            f"File too large: {size / (1024**2):.1f}MB (max: 100MB)",
        ),
    ]

    for condition, message in limits:
        if condition:
            return False, message

    return True, None


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_patterns = [
        "youtube.com/watch?v=",
        "youtu.be/",
        "youtube.com/embed/",
        "youtube.com/v=",
    ]
    url_lower = url.lower()
    return url_lower.startswith(("http://", "https://")) and any(
        pattern in url_lower for pattern in youtube_patterns
    )


def is_url(text: str) -> bool:
    """Check if string is a URL"""
    return text.startswith(("http://", "https://"))


def is_text_content(text: str, original_source: Union[str, Path]) -> bool:
    """Determine if string is text content vs URL/path"""
    # If it's a Path object, it's definitely a file path
    if isinstance(original_source, Path):
        return False

    # Heuristic: if it has newlines or is long, likely text content
    if "\n" in text or len(text) > 200:
        return True

    if is_url(text):
        return False

    # Check if it looks like a file path
    try:
        path = Path(text)
        if path.exists():
            return False

        # Even if path doesn't exist, check if it has a file extension
        # This handles cases where relative paths don't exist from current directory
        if path.suffix:
            extension = path.suffix.lower()
            # Use existing comprehensive extension checking
            if extension in ALL_SUPPORTED_EXTENSIONS:
                return False

    except (OSError, ValueError):
        pass

    return True  # Default to text content
