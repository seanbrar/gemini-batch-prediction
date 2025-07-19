"""Content extraction from various file types with multimodal support"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import GeminiBatchError
from . import utils
from .scanner import FileInfo, FileType

log = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Container for extracted content and metadata"""

    content: str  # For text files: actual content. For multimodal: empty string
    metadata: dict[str, Any]
    file_info: FileInfo
    extraction_method: str
    file_path: Path | None = None  # For API upload files: actual file path

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
    def media_type(self) -> str | None:
        """Get the media type for multimodal files"""
        return self.metadata.get("media_type")

    @property
    def processing_strategy(self) -> str:
        """How this content should be processed - 'upload', 'inline', 'text_only', 'url'"""
        if self.extraction_method in ["youtube_api", "pdf_url_api"]:
            return "url"  # Direct API (YouTube)
        if self.extraction_method == "arxiv_pdf_download":
            return "upload"  # Download + Upload (arXiv PDFs)
        if self.requires_api_upload:
            return "upload"
        if self.content:
            return "text_only"
        return "inline"


class BaseExtractor(ABC):
    """Base class for content extractors"""

    def __init__(self, max_size: int = utils.MAX_TEXT_SIZE):
        self.max_size = max_size

    @abstractmethod
    def can_extract(self, file_info: FileInfo) -> bool:
        """Check if this extractor can handle the file type"""

    @abstractmethod
    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Extract content from the file"""

    def _validate_file_size(self, file_info: FileInfo) -> None:
        """Validate file size against limits"""
        if file_info.size > self.max_size:
            raise GeminiBatchError(
                f"File too large: {file_info.size / (1024**2):.1f}MB "
                f"(max: {self.max_size / (1024**2):.1f}MB)",
            )

    def _get_base_metadata(self, file_info: FileInfo) -> dict[str, Any]:
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
        """Prepare media file for processing (inline or API upload based on size)"""
        log.debug(
            "Processing %s file: %s (%d bytes)",
            file_info.file_type.value,
            file_info.path,
            file_info.size,
        )

        self._validate_file_size(file_info)

        # Get base metadata (includes size-based upload decision) and add media-specific info
        metadata = self._get_base_metadata(file_info)
        metadata.update(
            {
                "media_type": self.MEDIA_TYPE_MAP.get(file_info.file_type, "document"),
            },
        )

        extracted = ExtractedContent(
            content="",  # No text content - file will be processed
            metadata=metadata,
            file_info=file_info,
            file_path=file_info.path,
            extraction_method="media",
        )

        return extracted


class URLExtractor(BaseExtractor):
    """Extract content from URLs - supports arXiv PDFs only"""

    def __init__(self, timeout: int = 30, max_size: int = utils.MAX_FILE_SIZE):
        super().__init__(max_size)
        self.timeout = timeout
        self._content_cache = {}  # Cache downloaded content to avoid double downloads

    def can_extract(self, source: str) -> bool:
        """Check if source is a supported URL (arXiv PDFs only)"""
        if not utils.is_url(source):
            return False

        # Exclude YouTube URLs - they're handled by YouTubeExtractor
        if utils.is_youtube_url(source):
            return False

        # Only support arXiv PDF URLs
        return self._is_arxiv_pdf_url(source)

    def can_extract_source(self, source: str | Path) -> bool:
        """Check if source is a supported URL (arXiv PDFs only)"""
        return self.can_extract(str(source))

    def _is_arxiv_pdf_url(self, url: str) -> bool:
        """Check if URL is a supported arXiv PDF"""
        url_lower = url.lower()

        # Specific arXiv PDF patterns
        arxiv_patterns = ["arxiv.org/pdf/", "export.arxiv.org/pdf/"]

        return any(pattern in url_lower for pattern in arxiv_patterns)

    def _create_http_client(self) -> httpx.Client:
        """Create a configured HTTP client - centralized configuration"""
        return httpx.Client(timeout=self.timeout)

    def _cache_content(self, url: str, content: bytes) -> None:
        """Cache content for a URL - centralized caching logic"""
        self._content_cache[url] = content

    def _get_cached_content(self, url: str) -> bytes | None:
        """Get cached content if available"""
        return self._content_cache.get(url)

    def _make_http_request(self, url: str, method: str = "GET") -> httpx.Response:
        """Make HTTP request with centralized error handling"""
        try:
            with self._create_http_client() as client:
                if method.upper() == "HEAD":
                    response = client.head(url, follow_redirects=True)
                else:
                    response = client.get(url, follow_redirects=True)

                response.raise_for_status()
                return response
        except httpx.TimeoutException as e:
            raise GeminiBatchError(f"URL request timeout: {url}") from e
        except httpx.HTTPStatusError as e:
            raise GeminiBatchError(f"HTTP error {e.response.status_code}: {url}") from e
        except Exception as e:
            raise GeminiBatchError(f"Failed to fetch URL {url}: {e}") from e

    def _extract_mime_and_size_from_response(
        self,
        response: httpx.Response,
    ) -> tuple[str, int]:
        """Extract MIME type and content size from HTTP response"""
        mime_type = response.headers.get("content-type", "").split(";")[0]
        content_length = response.headers.get("content-length")

        if hasattr(response, "content"):
            # GET response - we have content
            content_size = len(response.content)
            return mime_type, content_size
        if content_length:
            # HEAD response with content-length
            return mime_type, int(content_length)
        # HEAD response without content-length - need to do GET
        return None, None

    def _get_content_metadata(self, url: str) -> tuple[str, int]:
        """Get content metadata efficiently using HEAD request first, with GET fallback"""
        # Check cache first
        cached_content = self._get_cached_content(url)
        if cached_content:
            mime_type = self._determine_mime_type_from_url(url)
            return mime_type, len(cached_content)

        # Try HEAD request first
        try:
            head_response = self._make_http_request(url, "HEAD")
            mime_type, content_size = self._extract_mime_and_size_from_response(
                head_response,
            )

            if mime_type and content_size is not None:
                return mime_type or self._determine_mime_type_from_url(
                    url,
                ), content_size
        except GeminiBatchError as e:
            # If HEAD fails with 405/501, fall back to GET
            if "405" in str(e) or "501" in str(e):
                pass  # Continue to GET request
            else:
                raise

        # Fallback to GET request
        get_response = self._make_http_request(url, "GET")
        mime_type, content_size = self._extract_mime_and_size_from_response(
            get_response,
        )

        # Cache the content since we downloaded it
        if hasattr(get_response, "content"):
            self._cache_content(url, get_response.content)

        return mime_type or self._determine_mime_type_from_url(url), content_size

    def _determine_mime_type_from_url(self, url: str) -> str:
        """Determine MIME type from URL using centralized utilities with minimal fallback"""
        try:
            url_path = Path(url)
            extension = url_path.suffix.lower()
            if extension in utils.EXTENSION_TO_MIME:
                return utils.EXTENSION_TO_MIME[extension]
        except Exception:
            pass

        # Minimal fallback for common cases not in utils
        return "application/octet-stream"

    def _is_large_pdf_url(self, url: str) -> bool:
        """Check if arXiv PDF URL is large enough to warrant Files API handling"""
        if not self._is_arxiv_pdf_url(url):
            return False

        try:
            head_response = self._make_http_request(url, "HEAD")
            content_length = head_response.headers.get("content-length")

            if content_length:
                size = int(content_length)
                return size > utils.FILES_API_THRESHOLD

            # If no content-length header, assume it might be large
            # ArXiv papers often don't include content-length
            return True

        except Exception:
            # If HEAD request fails, assume not large for safety
            return False

    def _validate_video_url(self, url: str):
        """Reject non-YouTube video URLs with helpful message"""
        video_indicators = ["video", "watch", "embed", "player"]
        if not utils.is_youtube_url(url) and any(
            indicator in url.lower() for indicator in video_indicators
        ):
            raise GeminiBatchError(
                f"Only YouTube videos are supported directly. "
                f"For other platforms, please download: {url}",
            )

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract arXiv PDF URLs for processing"""
        # Validate that this is a supported URL
        if not self._is_arxiv_pdf_url(url):
            raise GeminiBatchError(
                f"Unsupported URL: {url}. "
                f"Only arXiv PDF URLs (arxiv.org/pdf/) are supported. "
                f"For YouTube videos, use the direct URL.",
            )

        # Get content metadata
        mime_type, content_size = self._get_content_metadata(url)

        # Validate size
        if content_size > self.max_size:
            raise GeminiBatchError(
                f"PDF too large: {content_size / (1024**2):.1f}MB "
                f"(max: {self.max_size / (1024**2):.1f}MB)",
            )

        # Create FileInfo for arXiv PDF
        url_path = Path(url)
        file_info = FileInfo(
            path=url_path,
            file_type=FileType.PDF,
            size=content_size,
            extension=".pdf",
            name=f"arxiv_paper_{url_path.name.replace('/', '_')}",
            relative_path=url_path,
            mime_type="application/pdf",
        )

        # Determine if the file should be uploaded to the Files API
        requires_upload = utils.requires_files_api(content_size)

        metadata = {
            "url": url,
            "file_size": content_size,
            "file_size_mb": round(content_size / (1024 * 1024), 2),
            "mime_type": "application/pdf",
            "requires_api_upload": requires_upload,
            "processing_method": "files_api" if requires_upload else "inline",
            "source_type": "arxiv_pdf",
            "download_method": "httpx",
            "content_cached": url in self._content_cache,
        }

        return ExtractedContent(
            content="",  # No content - PDF will be downloaded when needed
            metadata=metadata,
            file_info=file_info,
            file_path=None,  # No local file path - URL will be downloaded
            extraction_method="arxiv_pdf_download",
        )

    def _extract_large_pdf_url(self, url: str) -> ExtractedContent:
        """Handle large PDF URLs that should be passed directly to API"""
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

    def extract_source(self, source: str | Path) -> ExtractedContent:
        """Extract content from URL source"""
        return self.extract_from_url(str(source))

    def get_cached_content(self, url: str) -> bytes | None:
        """Get cached content for a URL to avoid re-downloading"""
        return self._get_cached_content(url)

    def download_content_if_needed(self, url: str) -> bytes:
        """Download content only if not already cached"""
        cached_content = self._get_cached_content(url)
        if cached_content:
            return cached_content

        # Download and cache
        response = self._make_http_request(url, "GET")
        content_data = response.content
        self._cache_content(url, content_data)
        return content_data

    def extract(self, file_info: FileInfo) -> ExtractedContent:
        """Extract method for compatibility with BaseExtractor interface"""
        # This shouldn't be called directly for URLs, but provide fallback
        return self.extract_from_url(str(file_info.path))

    def _get_extension_from_url(self, url: str) -> str:
        """Extract file extension from URL"""
        try:
            url_path = Path(url)
            return url_path.suffix.lower()
        except Exception:
            return ""


class ContentExtractorManager:
    """Content extractor manager with automatic extractor selection"""

    def __init__(self, custom_extractors: list[BaseExtractor] | None = None):
        """Initialize with default extractors and optional custom ones"""
        self.extractors = [
            TextContentExtractor(),
            YouTubeExtractor(),
            TextExtractor(),
            MediaExtractor(),
            URLExtractor(),
        ]

        if custom_extractors:
            self.extractors.extend(custom_extractors)

    def process_source(self, source: str | Path) -> ExtractedContent:
        """Process any source type using the appropriate extractor"""
        # Try extractors that can handle source directly (string/Path)
        for extractor in self.extractors:
            if hasattr(
                extractor,
                "can_extract_source",
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
            if path.exists() and path.is_dir():
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
            f"(type: {file_info.file_type}, mime: {file_info.mime_type})",
        )

    def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from URL using URLExtractor"""
        url_extractor = URLExtractor()
        if url_extractor.can_extract(url):
            return url_extractor.extract_from_url(url)
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

    def can_extract_source(self, source: str | Path) -> bool:
        """Check if source is direct text content"""
        return utils.is_text_content(str(source), source)

    def extract_source(self, source: str | Path) -> ExtractedContent:
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


class YouTubeExtractor(BaseExtractor):
    """Handle YouTube URLs"""

    def can_extract(self, file_info: FileInfo) -> bool:
        """This extractor doesn't work with FileInfo"""
        return False

    def can_extract_source(self, source: str | Path) -> bool:
        """Check if source is a YouTube URL"""
        return utils.is_youtube_url(str(source))

    def extract_source(self, source: str | Path) -> ExtractedContent:
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
