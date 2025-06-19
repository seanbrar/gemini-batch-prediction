"""
Content extraction from various file types with multimodal support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

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


class URLExtractor(BaseExtractor):
    """Extract content from URLs by downloading and processing"""

    def __init__(self, timeout: int = 30, max_size: int = utils.MAX_FILE_SIZE):
        super().__init__(max_size)
        self.timeout = timeout
        self._content_cache = {}  # Cache downloaded content to avoid double downloads

    def can_extract(self, source: str) -> bool:
        """Check if source is a URL (non-YouTube)"""
        if not utils.is_url(source):
            return False

        # Exclude YouTube URLs - they're handled separately
        return not utils.is_youtube_url(source)

    def _validate_video_url(self, url: str):
        """Reject non-YouTube video URLs with helpful message"""
        video_indicators = ["video", "watch", "embed", "player"]
        if not utils.is_youtube_url(url) and any(
            indicator in url.lower() for indicator in video_indicators
        ):
            raise GeminiBatchError(
                f"Only YouTube videos are supported directly. "
                f"For other platforms, please download: {url}"
            )

    def _get_content_metadata_with_head(self, url: str) -> tuple[str, int]:
        """Use HEAD request to get content metadata without downloading"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.head(url, follow_redirects=True)
                response.raise_for_status()

                # Get MIME type and content length
                mime_type = response.headers.get("content-type", "").split(";")[0]
                content_length = response.headers.get("content-length")

                if content_length:
                    size = int(content_length)
                else:
                    # Fallback: do a GET request to determine size
                    get_response = client.get(url)
                    get_response.raise_for_status()
                    content_data = get_response.content
                    size = len(content_data)
                    mime_type = (
                        get_response.headers.get("content-type", "").split(";")[0]
                        or mime_type
                    )

                    # Cache the content since we downloaded it
                    self._content_cache[url] = content_data

                return mime_type or self._guess_mime_type_from_url(url), size

        except httpx.HTTPStatusError as e:
            # Fallback to GET if HEAD fails
            if e.response.status_code in [
                405,
                501,
            ]:  # Method not allowed/not implemented
                return self._fallback_to_get_request(url)
            raise

    def _fallback_to_get_request(self, url: str) -> tuple[str, int]:
        """Fallback to GET request when HEAD is not supported"""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            content_data = response.content

            # Cache the content
            self._content_cache[url] = content_data

            mime_type = response.headers.get("content-type", "").split(";")[0]
            return mime_type or self._guess_mime_type_from_url(url), len(content_data)

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from URL using efficient HEAD request first"""
        try:
            # Validate video URLs early
            self._validate_video_url(url)

            # Get metadata efficiently (HEAD request or cached GET)
            mime_type, content_size = self._get_content_metadata_with_head(url)

            # Validate size
            if content_size > self.max_size:
                raise GeminiBatchError(
                    f"URL content too large: {content_size / (1024**2):.1f}MB "
                    f"(max: {self.max_size / (1024**2):.1f}MB)"
                )

            # Determine file type from MIME type
            file_type, detected_mime = utils.determine_file_type(Path(url), mime_type)

            # Create synthetic FileInfo for URL content
            url_path = Path(url)
            file_info = FileInfo(
                path=url_path,
                file_type=file_type,
                size=content_size,
                extension=self._get_extension_from_url(url),
                name=url_path.name or "downloaded_content",
                relative_path=url_path,
                mime_type=detected_mime or mime_type,
            )

            # Create metadata
            metadata = {
                "url": url,
                "file_size": content_size,
                "file_size_mb": round(content_size / (1024 * 1024), 2),
                "mime_type": detected_mime or mime_type,
                "requires_api_upload": utils.requires_files_api(content_size),
                "processing_method": "files_api"
                if utils.requires_files_api(content_size)
                else "inline",
                "source_type": "url",
                "download_method": "httpx",
                "content_cached": url
                in self._content_cache,  # Track if we have cached content
            }

            return ExtractedContent(
                content="",  # No text content - raw bytes will be used
                metadata=metadata,
                file_info=file_info,
                file_path=None,  # No local file path for URLs
                extraction_method="url_download",
            )

        except httpx.TimeoutException as e:
            raise GeminiBatchError(f"URL request timeout: {url}") from e
        except httpx.HTTPStatusError as e:
            raise GeminiBatchError(f"HTTP error {e.response.status_code}: {url}") from e
        except Exception as e:
            raise GeminiBatchError(f"Failed to fetch URL {url}: {e}") from e

    def get_cached_content(self, url: str) -> Optional[bytes]:
        """Get cached content for a URL to avoid re-downloading"""
        return self._content_cache.get(url)

    def download_content_if_needed(self, url: str) -> bytes:
        """Download content only if not already cached"""
        if url in self._content_cache:
            return self._content_cache[url]

        # Download and cache
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            content_data = response.content
            self._content_cache[url] = content_data
            return content_data

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Extract method for compatibility with BaseExtractor interface"""
        # This shouldn't be called directly for URLs, but provide fallback
        return self.extract_from_url(str(file_info.path))

    def _guess_mime_type_from_url(self, url: str) -> str:
        """Guess MIME type from URL extension using centralized utilities"""
        # Try to use the centralized MIME detection
        try:
            url_path = Path(url)
            extension = url_path.suffix.lower()
            if extension in utils.EXTENSION_TO_MIME:
                return utils.EXTENSION_TO_MIME[extension]
        except Exception:
            pass

        # Fallback to simple guessing
        url_lower = url.lower()
        if url_lower.endswith(".pdf"):
            return "application/pdf"
        elif url_lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif url_lower.endswith(".png"):
            return "image/png"
        elif url_lower.endswith(".mp4"):
            return "video/mp4"
        elif url_lower.endswith(".mp3"):
            return "audio/mpeg"
        else:
            return "application/octet-stream"

    def _get_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL"""
        try:
            url_path = Path(url)
            return url_path.suffix.lower()
        except Exception:
            return ""


class ContentExtractorManager:
    """Content extractor manager with automatic extractor selection"""

    def __init__(self, custom_extractors: Optional[List[BaseExtractor]] = None):
        """Initialize with default extractors and optional custom ones"""
        self.extractors = [
            TextContentExtractor(),
            PDFURLExtractor(),  # Priority: handle large PDF URLs first
            YouTubeExtractor(),
            TextExtractor(),
            MediaExtractor(),
            URLExtractor(),  # Fallback: handle remaining URLs
        ]

        if custom_extractors:
            self.extractors.extend(custom_extractors)

    def process_source(self, source: Union[str, Path]) -> ExtractedContent:
        """Process any source type using the appropriate extractor"""
        # Try extractors that can handle source directly (string/Path)
        for extractor in self.extractors:
            if hasattr(
                extractor, "can_extract_source"
            ) and extractor.can_extract_source(source):
                try:
                    return extractor.extract_source(source)
                except Exception:
                    continue

        # Fall back to file-based extraction for Path objects
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                return self.extract_content_from_path(path)
            elif path.exists() and path.is_dir():
                # Return special marker for directory processing
                from .scanner import FileInfo, FileType

                return ExtractedContent(
                    content="",
                    metadata={"source_type": "directory", "directory_path": str(path)},
                    file_info=FileInfo(
                        path=path,
                        file_type=FileType.UNKNOWN,
                        size=0,
                        extension="",
                        name=path.name,
                        relative_path=path,
                        mime_type="application/directory",
                    ),
                    extraction_method="directory_scan",
                )

        raise GeminiBatchError(f"No extractor available for source: {source}")

    def extract_content_from_path(self, file_path: Path) -> ExtractedContent:
        """Extract content from file path (original method)"""
        from .scanner import DirectoryScanner

        scanner = DirectoryScanner()
        file_info = scanner._create_file_info(file_path, file_path.parent)
        return self.extract_content(file_info)

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

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from URL using URLExtractor"""
        url_extractor = URLExtractor()
        if url_extractor.can_extract(url):
            return url_extractor.extract_from_url(url)
        else:
            raise GeminiBatchError(f"URL not supported: {url}")

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


class TextContentExtractor(BaseExtractor):
    """Handle direct text content (strings that aren't URLs or file paths)"""

    def can_extract(self, file_info: FileInfo) -> bool:
        """This extractor doesn't work with FileInfo"""
        return False

    def can_extract_source(self, source: Union[str, Path]) -> bool:
        """Check if source is direct text content"""
        return utils.is_text_content(str(source), source)

    def extract_source(self, source: Union[str, Path]) -> ExtractedContent:
        """Extract text content directly"""
        text = str(source)
        from .scanner import FileInfo, FileType

        return ExtractedContent(
            content=text,
            metadata={"source_type": "text", "processing_method": "direct"},
            file_info=FileInfo(
                path=Path("text_content"),
                file_type=FileType.TEXT,
                size=len(text),
                extension="",
                name="text_content",
                relative_path=Path("text_content"),
                mime_type="text/plain",
            ),
            extraction_method="direct_text",
        )

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Not used - this extractor works with sources directly"""
        raise NotImplementedError("Use extract_source() instead")


class PDFURLExtractor(BaseExtractor):
    """Handle large PDF URLs that should be passed directly to Gemini API"""

    def __init__(self, timeout: int = 30):
        super().__init__()
        self.timeout = timeout

    def can_extract(self, file_info: FileInfo) -> bool:
        """This extractor doesn't work with FileInfo"""
        return False

    def can_extract_source(self, source: Union[str, Path]) -> bool:
        """Check if source is a large PDF URL"""
        if not utils.is_url(str(source)):
            return False

        url = str(source)

        # Quick check for obvious PDF URLs
        if url.lower().endswith(".pdf"):
            return self._is_large_pdf_url(url)

        # Check for common PDF hosting patterns
        pdf_indicators = ["arxiv.org/pdf", "pdf", ".pdf"]
        if any(indicator in url.lower() for indicator in pdf_indicators):
            return self._is_large_pdf_url(url)

        return False

    def _is_large_pdf_url(self, url: str) -> bool:
        """Check if PDF URL is large enough to warrant direct API handling"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.head(url, follow_redirects=True)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "").split(";")[0]
                if content_type != "application/pdf":
                    return False

                # Check size - if >20MB, handle directly via API
                content_length = response.headers.get("content-length")
                if content_length:
                    size = int(content_length)
                    return size > utils.FILES_API_THRESHOLD

                # If no content-length header, assume it might be large
                # ArXiv papers often don't include content-length
                return True

        except Exception:
            # If HEAD request fails, fall back to regular URL processing
            return False

    def extract_source(self, source: Union[str, Path]) -> ExtractedContent:
        """Extract large PDF URL for direct API processing"""
        url = str(source)
        from .scanner import FileInfo, FileType

        return ExtractedContent(
            content="",  # No content - will be handled by API
            metadata={
                "source_type": "pdf_url",
                "processing_method": "pdf_url_api",
                "url": url,
            },
            file_info=FileInfo(
                path=Path(url),
                file_type=FileType.PDF,
                size=0,
                extension=".pdf",
                name="pdf_from_url",
                relative_path=Path(url),
                mime_type="application/pdf",
            ),
            extraction_method="pdf_url_api",
        )

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Not used - this extractor works with sources directly"""
        raise NotImplementedError("Use extract_source() instead")


class YouTubeExtractor(BaseExtractor):
    """Handle YouTube URLs"""

    def can_extract(self, file_info: FileInfo) -> bool:
        """This extractor doesn't work with FileInfo"""
        return False

    def can_extract_source(self, source: Union[str, Path]) -> bool:
        """Check if source is a YouTube URL"""
        return utils.is_youtube_url(str(source))

    def extract_source(self, source: Union[str, Path]) -> ExtractedContent:
        """Extract YouTube URL for API processing"""
        url = str(source)
        from .scanner import FileInfo, FileType

        return ExtractedContent(
            content="",  # No content - will be handled by API
            metadata={
                "source_type": "youtube",
                "processing_method": "youtube_api",
                "url": url,
            },
            file_info=FileInfo(
                path=Path(url),
                file_type=FileType.VIDEO,
                size=0,
                extension="",
                name="youtube_video",
                relative_path=Path(url),
                mime_type="video/youtube",
            ),
            extraction_method="youtube_api",
        )

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Not used - this extractor works with sources directly"""
        raise NotImplementedError("Use extract_source() instead")
