"""
File operations for the Gemini Batch Framework
"""  # noqa: D200, D212, D415

import logging
from pathlib import Path
from typing import Any, Dict, Union  # noqa: UP035

from ..exceptions import FileError  # noqa: TID252
from . import utils
from .extractors import ContentExtractorManager, ExtractedContent
from .scanner import DirectoryScanner, FileInfo, FileType

log = logging.getLogger(__name__)


class FileOperations:
    """Low-level file and directory operations"""  # noqa: D415

    def __init__(self):  # noqa: ANN204
        """Initialize file operations components"""  # noqa: D415
        self.scanner = DirectoryScanner()
        self.extractor_manager = ContentExtractorManager()

    def _validate_path(
        self, file_path: Union[str, Path], must_be_file: bool = True  # noqa: COM812, FBT001, FBT002, UP007
    ) -> Path:
        """Validate path exists and is of the correct type"""  # noqa: D415
        path = Path(file_path)

        if not path.exists():
            raise FileError(f"Path not found: {path}")  # noqa: EM102, TRY003

        if must_be_file and not path.is_file():
            raise FileError(f"Path is not a file: {path}")  # noqa: EM102, TRY003
        elif not must_be_file and not path.is_dir():  # noqa: RET506
            raise FileError(f"Path is not a directory: {path}")  # noqa: EM102, TRY003

        return path

    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:  # noqa: UP007
        """Extract content from a file for processing"""  # noqa: D415
        path = self._validate_path(file_path, must_be_file=True)

        # Create FileInfo for the extractor
        file_info = self.scanner._create_file_info(path, path.parent)  # noqa: SLF001

        # Extract content using the manager
        return self.extractor_manager.extract_content(file_info)

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from URL using URLExtractor"""  # noqa: D415
        return self.extractor_manager.extract_from_url(url)

    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:  # noqa: UP006, UP007
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
            file_path=path, mime_type=mime_type, extension=suffix, file_type=file_type  # noqa: COM812
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
        self, directory_path: Union[str, Path], **kwargs  # noqa: ANN003, COM812, UP007
    ) -> Dict[FileType, list[FileInfo]]:  # noqa: UP006
        """Scan directory for supported files"""  # noqa: D415
        path = self._validate_path(directory_path, must_be_file=False)
        return self.scanner.scan_directory(path, **kwargs)

    def get_file_info(
        self, file_path: Union[str, Path], root_dir: Union[str, Path] = None  # noqa: COM812, RUF013, UP007
    ) -> FileInfo:
        """Get FileInfo object for a single file"""  # noqa: D415
        path = self._validate_path(file_path, must_be_file=True)

        root_dir = Path(root_dir) if root_dir else path.parent
        return self.scanner._create_file_info(path, root_dir)  # noqa: SLF001

    def get_directory_summary(
        self, directory_path: Union[str, Path], **kwargs  # noqa: ANN003, COM812, UP007
    ) -> Dict[str, Any]:  # noqa: UP006
        """Get a summary of files in a directory"""  # noqa: D415
        scan_results = self.scan_directory(directory_path, **kwargs)
        return self.scanner.get_summary(scan_results)

    def process_source(self, source: Union[str, Path]) -> ExtractedContent:  # noqa: UP007
        """Process any source type (text, URLs, files, directories) using appropriate extractors"""  # noqa: D415, E501
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
