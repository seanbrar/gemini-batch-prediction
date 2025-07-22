"""File operations for the Gemini Batch Framework"""  # noqa: D415

import logging
from pathlib import Path
from typing import Any

from ..exceptions import FileError
from . import utils
from .extractors import ContentExtractorManager, ExtractedContent
from .scanner import DirectoryScanner, FileInfo, FileType

log = logging.getLogger(__name__)


class FileOperations:
    """Low-level file and directory operations"""  # noqa: D415

    def __init__(self):
        """Initialize file operations components"""  # noqa: D415
        self.scanner = DirectoryScanner()
        self.extractor_manager = ContentExtractorManager()

    def _validate_path(
        self,
        file_path: str | Path,
        must_be_file: bool = True,  # noqa: FBT001, FBT002
    ) -> Path:
        """Validate path exists and is of the correct type"""  # noqa: D415
        path = Path(file_path)

        if not path.exists():
            raise FileError(f"Path not found: {path}")

        if must_be_file and not path.is_file():
            raise FileError(f"Path is not a file: {path}")
        if not must_be_file and not path.is_dir():
            raise FileError(f"Path is not a directory: {path}")

        return path

    def extract_content(self, file_path: str | Path) -> ExtractedContent:
        """Extract content from a file for processing"""  # noqa: D415
        path = self._validate_path(file_path, must_be_file=True)

        # Create FileInfo for the extractor
        file_info = self.scanner._create_file_info(path, path.parent)

        # Extract content using the manager
        return self.extractor_manager.extract_content(file_info)

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from URL using URLExtractor"""  # noqa: D415
        return self.extractor_manager.extract_from_url(url)

    def validate_file(self, file_path: str | Path) -> dict[str, Any]:
        """Validate a file and return metadata"""  # noqa: D415
        path = self._validate_path(file_path, must_be_file=True)

        # Get basic file info
        size = path.stat().st_size
        suffix = path.suffix.lower()

        # Determine MIME type and file type using centralized utilities
        mime_type = utils.get_mime_type(path)
        file_type, _ = utils.determine_file_type(path, mime_type)

        # Check if supported using centralized function
        supported = utils.is_supported_file(
            file_path=path,
            mime_type=mime_type,
            extension=suffix,
            file_type=file_type,
        )

        return {
            "path": str(path),
            "name": path.name,
            "size": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "mime_type": mime_type,
            "file_type": file_type.value,
            "extension": suffix,
            "supported": supported,
            "processing_method": "files_api"
            if utils.requires_files_api(size)
            else "inline",
        }

    def scan_directory(
        self,
        directory_path: str | Path,
        **kwargs,
    ) -> dict[FileType, list[FileInfo]]:
        """Scan directory for supported files"""  # noqa: D415
        path = self._validate_path(directory_path, must_be_file=False)
        return self.scanner.scan_directory(path, **kwargs)

    def get_file_info(
        self,
        file_path: str | Path,
        root_dir: str | Path = None,  # noqa: RUF013
    ) -> FileInfo:
        """Get FileInfo object for a single file"""  # noqa: D415
        path = self._validate_path(file_path, must_be_file=True)

        root_dir = Path(root_dir) if root_dir else path.parent
        return self.scanner._create_file_info(path, root_dir)

    def get_directory_summary(
        self,
        directory_path: str | Path,
        **kwargs,
    ) -> dict[str, Any]:
        """Get a summary of files in a directory"""  # noqa: D415
        scan_results = self.scan_directory(directory_path, **kwargs)
        return self.scanner.get_summary(scan_results)

    def process_source(self, source: str | Path) -> ExtractedContent:
        """Process any source type (text, URLs, files, directories) using appropriate extractors"""  # noqa: D415
        result = self.extractor_manager.process_source(source)
        log.debug(
            "Processed source '%s': %s -> %s",
            source,
            result.extraction_method,
            result.file_info.file_type.value,
        )
        return result

    def is_multimodal_content(self, extracted_content: ExtractedContent) -> bool:
        """Check if extracted content is multimodal (PDF, image, video, audio)"""  # noqa: D415
        multimodal_types = {
            FileType.PDF,
            FileType.IMAGE,
            FileType.VIDEO,
            FileType.AUDIO,
        }
        return extracted_content.file_info.file_type in multimodal_types
