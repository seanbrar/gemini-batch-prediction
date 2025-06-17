"""
Content extraction from various file types with multimodal support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..exceptions import GeminiBatchError
from . import utils
from .scanner import FileInfo, FileType


@dataclass
class ExtractedContent:
    """Container for extracted content and metadata"""

    content: str  # For text files: actual content. For multimodal: empty string
    metadata: Dict[str, Any]
    file_info: FileInfo
    extraction_method: str
    file_path: Optional[Path] = None  # For API upload files: actual file path

    @property
    def size(self) -> int:
        """Size of extracted content in characters"""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Approximate word count"""
        return len(self.content.split())

    @property
    def requires_api_upload(self) -> bool:
        """Check if this content requires API upload"""
        return self.metadata.get("requires_api_upload", False)

    @property
    def media_type(self) -> Optional[str]:
        """Get the media type for multimodal files"""
        return self.metadata.get("media_type")


class BaseExtractor(ABC):
    """Base class for content extractors"""

    def __init__(self, max_size: int = utils.MAX_TEXT_SIZE):
        self.max_size = max_size

    @abstractmethod
    def can_extract(self, file_info: FileInfo) -> bool:
        """Check if this extractor can handle the file type"""
        pass

    @abstractmethod
    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Extract content from the file"""
        pass

    def _validate_file_size(self, file_info: FileInfo) -> None:
        """Validate file size against limits"""
        if file_info.size > self.max_size:
            raise GeminiBatchError(
                f"File too large: {file_info.size / (1024**2):.1f}MB "
                f"(max: {self.max_size / (1024**2):.1f}MB)"
            )

    def _get_base_metadata(self, file_info: FileInfo) -> Dict[str, Any]:
        """Get essential metadata for API submission"""
        return {
            "file_size": file_info.size,
            "file_size_mb": round(file_info.size / (1024 * 1024), 2),
            "requires_api_upload": utils.requires_files_api(file_info.size),
            "processing_method": "files_api"
            if utils.requires_files_api(file_info.size)
            else "inline",
            "mime_type": file_info.mime_type,
        }


class TextExtractor(BaseExtractor):
    """Prepare text files for Gemini API submission"""

    def can_extract(self, file_info: FileInfo) -> bool:
        """Check if file is a text file"""
        return utils.is_text_file(file_info.file_type, file_info.mime_type)

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Prepare text file for API submission"""
        self._validate_file_size(file_info)

        return ExtractedContent(
            content="",  # No local content extraction - let Gemini handle it
            metadata=self._get_base_metadata(file_info),
            file_info=file_info,
            file_path=file_info.path,
            extraction_method="gemini_native",
        )


class MediaExtractor(BaseExtractor):
    """Handle non-text files (images, videos, audio, PDFs) for API upload"""

    # Media type mappings for API submission
    MEDIA_TYPE_MAP = {
        FileType.IMAGE: "image",
        FileType.VIDEO: "video",
        FileType.AUDIO: "audio",
        FileType.PDF: "pdf",
    }

    def can_extract(self, file_info: FileInfo) -> bool:
        """Check if file is a supported media type (non-text)"""
        return file_info.file_type in {
            FileType.PDF,
            FileType.IMAGE,
            FileType.VIDEO,
            FileType.AUDIO,
        }

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Prepare media file for API upload"""
        self._validate_file_size(file_info)

        # Get base metadata and add media-specific info
        metadata = self._get_base_metadata(file_info)
        metadata.update(
            {
                "requires_api_upload": True,  # Media files always require API upload
                "media_type": self.MEDIA_TYPE_MAP.get(file_info.file_type, "document"),
            }
        )

        return ExtractedContent(
            content="",  # No text content - file will be uploaded to API
            metadata=metadata,
            file_info=file_info,
            file_path=file_info.path,
            extraction_method="media",
        )


class ContentExtractorManager:
    """Content extractor manager with automatic extractor selection"""

    def __init__(self, custom_extractors: Optional[List[BaseExtractor]] = None):
        """Initialize with default extractors and optional custom ones"""
        self.extractors = [TextExtractor(), MediaExtractor()]

        if custom_extractors:
            self.extractors.extend(custom_extractors)

    def extract_content(self, file_info: FileInfo) -> ExtractedContent:
        """Extract content using the first capable extractor"""
        for extractor in self.extractors:
            if extractor.can_extract(file_info):
                try:
                    return extractor.extract(file_info)
                except Exception:
                    # If this extractor fails, try the next one
                    continue

        # No extractor could handle the file
        raise GeminiBatchError(
            f"No extractor available for file: {file_info.path} "
            f"(type: {file_info.file_type}, mime: {file_info.mime_type})"
        )

    def register_extractor(self, extractor: BaseExtractor, priority: int = -1) -> None:
        """Add a custom extractor to the manager with optional priority"""
        if extractor not in self.extractors:
            if 0 <= priority < len(self.extractors):
                self.extractors.insert(priority, extractor)
            else:
                self.extractors.append(extractor)

    def get_supported_types(self) -> set[FileType]:
        """Get all file types supported by registered extractors"""
        return {
            FileType.TEXT,
            FileType.PDF,
            FileType.IMAGE,
            FileType.VIDEO,
            FileType.AUDIO,
        }
