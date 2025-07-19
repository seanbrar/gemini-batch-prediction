"""
File discovery and categorization for batch processing
"""  # noqa: D200, D212, D415

from dataclasses import dataclass
from enum import Enum
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Union  # noqa: UP035

from ..constants import SCANNER_MAX_SIZE  # noqa: TID252
from ..exceptions import GeminiBatchError  # noqa: TID252
from . import utils


class FileType(Enum):
    """File types supported by Gemini API"""  # noqa: D415

    TEXT = "text"  # JavaScript, Python, TXT, HTML, CSS, Markdown, CSV, XML, RTF
    PDF = "pdf"  # PDF documents (handled natively by Gemini)
    IMAGE = "image"  # JPEG, PNG, GIF, BMP, TIFF, SVG, WebP
    VIDEO = "video"  # MP4, AVI, MOV, WMV, FLV, WebM, MKV
    AUDIO = "audio"  # MP3, WAV, FLAC, AAC, OGG, WMA
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Information about a discovered file"""  # noqa: D415

    path: Path
    file_type: FileType
    size: int
    extension: str
    name: str
    relative_path: Path
    mime_type: Optional[str] = None  # noqa: UP045


class DirectoryScanner:
    """File discovery and categorization with filtering capabilities"""  # noqa: D415

    # Default exclusions
    DEFAULT_EXCLUDE_DIRS = {  # noqa: RUF012
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

    DEFAULT_EXCLUDE_FILES = {  # noqa: RUF012
        ".DS_Store",
        "Thumbs.db",
        ".gitignore",
        ".gitkeep",
        ".env",
        ".env.local",
        ".env.example",
    }

    def __init__(  # noqa: ANN204, PLR0913
        self,
        max_file_size: int = SCANNER_MAX_SIZE,
        include_types: Optional[Set[FileType]] = None,  # noqa: UP006, UP045
        exclude_types: Optional[Set[FileType]] = None,  # noqa: UP006, UP045
        exclude_dirs: Optional[Set[str]] = None,  # noqa: UP006, UP045
        exclude_files: Optional[Set[str]] = None,  # noqa: UP006, UP045
        include_patterns: Optional[List[str]] = None,  # noqa: UP006, UP045
        exclude_patterns: Optional[List[str]] = None,  # noqa: UP006, UP045
        use_magic: bool = True,  # noqa: FBT001, FBT002
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
        """  # noqa: D212, D415
        self.max_file_size = max_file_size
        self.include_types = include_types
        self.exclude_types = exclude_types or set()
        self.use_magic = use_magic

        # Combine default and custom exclusions
        self.exclude_dirs = self.DEFAULT_EXCLUDE_DIRS | (exclude_dirs or set())
        self.exclude_files = self.DEFAULT_EXCLUDE_FILES | (exclude_files or set())

        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def _get_file_type(self, file_path: Path) -> tuple[FileType, Optional[str]]:  # noqa: UP045
        """Determine file type using centralized utilities"""  # noqa: D415
        mime_type = utils.get_mime_type(file_path, self.use_magic)
        file_type, detected_mime = utils.determine_file_type(file_path, mime_type)
        return file_type, detected_mime

    def _should_exclude_dir(self, dir_name: str) -> bool:
        """Check if directory should be excluded"""  # noqa: D415
        return dir_name in self.exclude_dirs

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded based on various criteria"""  # noqa: D415
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
        """Check if file passes include/exclude pattern filters"""  # noqa: D415
        # If include patterns exist, file must match at least one
        if self.include_patterns and not self._matches_patterns(
            file_path, self.include_patterns  # noqa: COM812
        ):
            return False

        # If exclude patterns exist, file must not match any
        return not (
            self.exclude_patterns
            and self._matches_patterns(file_path, self.exclude_patterns)
        )

    def _matches_patterns(self, file_path: Path, patterns: List[str]) -> bool:  # noqa: UP006
        """Check if file matches any of the given patterns"""  # noqa: D415
        if not patterns:
            return False

        filename = file_path.name
        path_str = str(file_path)

        return any(
            fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(path_str, pattern)
            for pattern in patterns
        )

    def scan_directory(
        self, directory: Union[str, Path], recursive: bool = True  # noqa: COM812, FBT001, FBT002, UP007
    ) -> Dict[FileType, List[FileInfo]]:  # noqa: UP006
        """Scan directory and return categorized files"""  # noqa: D415
        directory = Path(directory)

        if not directory.exists():
            raise GeminiBatchError(f"Directory does not exist: {directory}")  # noqa: EM102, TRY003

        if not directory.is_dir():
            raise GeminiBatchError(f"Path is not a directory: {directory}")  # noqa: EM102, TRY003

        categorized_files = {file_type: [] for file_type in FileType}

        try:
            scanner_method = (
                self._scan_recursive if recursive else self._scan_single_level
            )
            scanner_method(directory, directory, categorized_files)
        except PermissionError as e:
            raise GeminiBatchError(f"Permission denied accessing directory: {e}") from e  # noqa: EM102, TRY003
        except OSError as e:
            raise GeminiBatchError(f"Error scanning directory: {e}") from e  # noqa: EM102, TRY003

        # Remove empty categories
        return {k: v for k, v in categorized_files.items() if v}

    def _scan_recursive(  # noqa: ANN202
        self,
        current_dir: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],  # noqa: UP006
    ):
        """Recursively scan directories"""  # noqa: D415
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

    def _scan_single_level(  # noqa: ANN202
        self,
        directory: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],  # noqa: UP006
    ):
        """Scan single directory level"""  # noqa: D415
        for item in directory.iterdir():
            if item.is_file():
                self._process_file(item, root_dir, categorized_files)

    def _process_file(  # noqa: ANN202
        self,
        file_path: Path,
        root_dir: Path,
        categorized_files: Dict[FileType, List[FileInfo]],  # noqa: UP006
    ):
        """Process a single file if it passes filters"""  # noqa: D415
        if not self._should_exclude_file(file_path):
            file_info = self._create_file_info(file_path, root_dir)
            categorized_files[file_info.file_type].append(file_info)

    def _create_file_info(self, file_path: Path, root_dir: Path) -> FileInfo:
        """Create FileInfo object from path with MIME type detection"""  # noqa: D415
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
        self, scan_results: Dict[FileType, List[FileInfo]]  # noqa: COM812, UP006
    ) -> Dict[str, int]:  # noqa: UP006
        """Generate summary statistics from scan results"""  # noqa: D415
        summary = {"total_files": 0, "total_size": 0}

        for file_type, files in scan_results.items():
            count = len(files)
            size = sum(f.size for f in files)

            summary[f"{file_type.value}_count"] = count
            summary[f"{file_type.value}_size"] = size
            summary["total_files"] += count
            summary["total_size"] += size

        return summary
