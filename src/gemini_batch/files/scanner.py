"""
File discovery and categorization for batch processing
"""

from dataclasses import dataclass
from enum import Enum
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ..constants import SCANNER_MAX_SIZE
from ..exceptions import GeminiBatchError
from . import utils


class FileType(Enum):
    """File types supported by Gemini API"""

    TEXT = "text"  # JavaScript, Python, TXT, HTML, CSS, Markdown, CSV, XML, RTF
    PDF = "pdf"  # PDF documents (handled natively by Gemini)
    IMAGE = "image"  # JPEG, PNG, GIF, BMP, TIFF, SVG, WebP
    VIDEO = "video"  # MP4, AVI, MOV, WMV, FLV, WebM, MKV
    AUDIO = "audio"  # MP3, WAV, FLAC, AAC, OGG, WMA
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Information about a discovered file"""

    path: Path
    file_type: FileType
    size: int
    extension: str
    name: str
    relative_path: Path
    mime_type: Optional[str] = None


class DirectoryScanner:
    """File discovery and categorization with filtering capabilities"""

    # Default exclusions
    DEFAULT_EXCLUDE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "build",
        "dist",
        ".tox",
        ".coverage",
    }

    DEFAULT_EXCLUDE_FILES = {
        ".DS_Store",
        "Thumbs.db",
        ".gitignore",
        ".gitkeep",
        ".env",
        ".env.local",
        ".env.example",
    }

    def __init__(
        self,
        max_file_size: int = SCANNER_MAX_SIZE,
        include_types: Optional[Set[FileType]] = None,
        exclude_types: Optional[Set[FileType]] = None,
        exclude_dirs: Optional[Set[str]] = None,
        exclude_files: Optional[Set[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        use_magic: bool = True,
    ):
        """
        Initialize directory scanner with filtering options

        Args:
            max_file_size: Maximum file size in bytes
            include_types: File types to include (None = all supported)
            exclude_types: File types to exclude
            exclude_dirs: Directory names to exclude
            exclude_files: File names to exclude
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            use_magic: Whether to use python-magic for content-based MIME detection
        """
        self.max_file_size = max_file_size
        self.include_types = include_types
        self.exclude_types = exclude_types or set()
        self.use_magic = use_magic

        # Combine default and custom exclusions
        self.exclude_dirs = self.DEFAULT_EXCLUDE_DIRS | (exclude_dirs or set())
        self.exclude_files = self.DEFAULT_EXCLUDE_FILES | (exclude_files or set())

        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def _get_file_type(self, file_path: Path) -> tuple[FileType, Optional[str]]:
        """Determine file type using centralized utilities"""
        mime_type = utils.get_mime_type(file_path, self.use_magic)
        file_type, detected_mime = utils.determine_file_type(file_path, mime_type)
        return file_type, detected_mime

    def _should_exclude_dir(self, dir_name: str) -> bool:
        """Check if directory should be excluded"""
        return dir_name in self.exclude_dirs

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded based on various criteria"""
        # Check file name exclusions
        if file_path.name in self.exclude_files:
            return True

        # Check size limits
        try:
            if file_path.stat().st_size > self.max_file_size:
                return True
        except OSError:
            return True  # Can't access file

        # Check file type exclusions
        file_type, _ = self._get_file_type(file_path)
        if file_type in self.exclude_types:
            return True

        # Check include types filter
        if self.include_types and file_type not in self.include_types:
            return True

        # Check patterns
        return not self._passes_pattern_filters(file_path)

    def _passes_pattern_filters(self, file_path: Path) -> bool:
        """Check if file passes include/exclude pattern filters"""
        # If include patterns exist, file must match at least one
        if self.include_patterns and not self._matches_patterns(
            file_path, self.include_patterns
        ):
            return False

        # If exclude patterns exist, file must not match any
        return not (
            self.exclude_patterns
            and self._matches_patterns(file_path, self.exclude_patterns)
        )

    def _matches_patterns(self, file_path: Path, patterns: List[str]) -> bool:
        """Check if file matches any of the given patterns"""
        if not patterns:
            return False

        filename = file_path.name
        path_str = str(file_path)

        return any(
            fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path_str, pattern)
            for pattern in patterns
        )

    def scan_directory(
        self, directory: Union[str, Path], recursive: bool = True
    ) -> Dict[FileType, List[FileInfo]]:
        """Scan directory and return categorized files"""
        directory = Path(directory)

        if not directory.exists():
            raise GeminiBatchError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise GeminiBatchError(f"Path is not a directory: {directory}")

        categorized_files = {file_type: [] for file_type in FileType}

        try:
            scanner_method = (
                self._scan_recursive if recursive else self._scan_single_level
            )
            scanner_method(directory, directory, categorized_files)
        except PermissionError as e:
            raise GeminiBatchError(f"Permission denied accessing directory: {e}") from e
        except OSError as e:
            raise GeminiBatchError(f"Error scanning directory: {e}") from e

        # Remove empty categories
        return {k: v for k, v in categorized_files.items() if v}

    def _scan_recursive(
        self,
        current_dir: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],
    ):
        """Recursively scan directories"""
        try:
            for item in current_dir.iterdir():
                if item.is_dir():
                    if not self._should_exclude_dir(item.name):
                        self._scan_recursive(item, root_dir, categorized_files)
                elif item.is_file():
                    self._process_file(item, root_dir, categorized_files)
        except PermissionError:
            # Skip directories we can't access
            pass

    def _scan_single_level(
        self,
        directory: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],
    ):
        """Scan single directory level"""
        for item in directory.iterdir():
            if item.is_file():
                self._process_file(item, root_dir, categorized_files)

    def _process_file(
        self,
        file_path: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],
    ):
        """Process a single file if it passes filters"""
        if not self._should_exclude_file(file_path):
            file_info = self._create_file_info(file_path, root_dir)
            categorized_files[file_info.file_type].append(file_info)

    def _create_file_info(self, file_path: Path, root_dir: Path) -> FileInfo:
        """Create FileInfo object from path with MIME type detection"""
        stat = file_path.stat()
        file_type, mime_type = self._get_file_type(file_path)

        return FileInfo(
            path=file_path,
            file_type=file_type,
            size=stat.st_size,
            extension=file_path.suffix.lower(),
            name=file_path.name,
            relative_path=file_path.relative_to(root_dir),
            mime_type=mime_type,
        )

    def get_summary(
        self, scan_results: Dict[FileType, List[FileInfo]]
    ) -> Dict[str, int]:
        """Generate summary statistics from scan results"""
        summary = {"total_files": 0, "total_size": 0}

        for file_type, files in scan_results.items():
            count = len(files)
            size = sum(f.size for f in files)

            summary[f"{file_type.value}_count"] = count
            summary[f"{file_type.value}_size"] = size
            summary["total_files"] += count
            summary["total_size"] += size

        return summary
