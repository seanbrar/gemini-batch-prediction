"""Content processing orchestration - converts multiple mixed sources
into processed ExtractedContent objects for API consumption
"""  # noqa: D205, D415

import io
import logging
from pathlib import Path
from typing import Any

from google.genai import types
import httpx

from gemini_batch.constants import LARGE_TEXT_THRESHOLD

from ..exceptions import APIError
from ..files import FileOperations
from ..files.extractors import ExtractedContent
from .file_upload_manager import FileUploadManager


def _flatten_sources(sources: str | Path | list) -> list[str | Path]:
    """Flatten nested source lists to avoid deep recursion in processing"""  # noqa: D415
    if not isinstance(sources, list):
        return [sources]

    flattened = []
    for item in sources:
        if item is None:
            continue  # Skip None values
        if isinstance(item, list):
            flattened.extend(_flatten_sources(item))
        else:
            flattened.append(item)
    return flattened


log = logging.getLogger(__name__)


class ContentProcessor:
    """Content orchestration layer - handles multiple mixed sources and batch processing.

    Responsibilities:
    - Convert multiple sources (files, directories, URLs) into ExtractedContent objects
    - Handle directory expansion and flattening
    - Coordinate with FileOperations for low-level file handling
    - Aggregate errors from batch processing
    - Create API parts from ExtractedContent (coordination layer)

    This class serves as an abstraction layer between GeminiClient (high-level API)
    and FileOperations (low-level file handling), preventing GeminiClient from
    needing to handle complex source orchestration logic.
    """

    def __init__(self):  # noqa: D107
        self.file_ops = FileOperations()
        # Lazy initialization of file upload manager
        self._file_upload_manager: FileUploadManager | None = None

    def _get_file_upload_manager(self, client: Any) -> FileUploadManager:
        """Lazy initialization of file upload manager"""  # noqa: D415
        if self._file_upload_manager is None:
            self._file_upload_manager = FileUploadManager(client)
        return self._file_upload_manager

    def process(
        self,
        content: str | Path | list[str | Path],
        client: Any | None = None,
    ) -> list[types.Part]:
        """Convert content to API parts - unified interface for GeminiClient.

        This method combines process_content() and create_api_parts() to provide
        a simple interface for converting any content source into API-ready parts.
        """
        extracted_contents = self.process_content(content)
        return self.create_api_parts(
            extracted_contents,
            cache_enabled=False,
            client=client,
        )

    def process_content(
        self,
        source: str | Path | list[str | Path],
    ) -> list[ExtractedContent]:
        """Convert any content source into list of ExtractedContent for API consumption"""  # noqa: D415
        # Flatten nested lists first to avoid deep recursion
        flattened_sources = _flatten_sources(source)

        extracted_contents = []
        for single_source in flattened_sources:
            # Process each flattened source (might expand directories)
            source_extracts = self._process_single_source(single_source)
            extracted_contents.extend(source_extracts)

        log.info(
            "Processed %d sources into %d content items",
            len(flattened_sources),
            len(extracted_contents),
        )
        return extracted_contents

    def is_multimodal_content(self, extracted_content: ExtractedContent) -> bool:
        """Check if content is multimodal by delegating to FileOperations"""  # noqa: D415
        return self.file_ops.is_multimodal_content(extracted_content)

    def _process_single_source(
        self,
        source: str | Path,
    ) -> list[ExtractedContent]:
        """Process a single content source to list of ExtractedContent (directories expand to multiple)"""  # noqa: D415
        try:
            # Get initial extraction from file operations
            extracted_content = self.file_ops.process_source(source)

            # Check if this is a directory marker that needs expansion
            if extracted_content.extraction_method == "directory_scan":
                return self._expand_directory(extracted_content)
            return [extracted_content]

        except Exception as e:
            log.error("Error processing source '%s': %s", source, e, exc_info=True)
            # Let the error bubble up rather than trying to create invalid ExtractedContent
            # The caller (GeminiClient) can handle this more appropriately
            raise RuntimeError(f"Error processing {source}: {e}") from e

    def _expand_directory(
        self,
        directory_extract: ExtractedContent,
    ) -> list[ExtractedContent]:
        """Expand a directory marker into individual file extractions"""  # noqa: D415
        directory_path = Path(directory_extract.metadata["directory_path"])
        log.debug("Expanding directory: %s", directory_path)

        try:
            # Use file operations to scan the directory
            categorized_files = self.file_ops.scan_directory(directory_path)

            # Flatten all files into a single list
            all_files = []
            for file_type, files in categorized_files.items():  # noqa: B007
                all_files.extend(files)

            if not all_files:
                # Return empty directory marker
                return [
                    ExtractedContent(
                        content=f"Directory {directory_path} contains no processable files.",
                        extraction_method="empty_directory",
                        metadata={
                            "directory_path": str(directory_path),
                            "file_count": 0,
                            "requires_api_upload": False,
                        },
                        file_info=directory_extract.file_info,
                    ),
                ]

            # Process each file individually
            file_extracts = []
            for file_info in all_files:
                try:
                    file_extract = self.file_ops.extract_content(file_info.path)
                    file_extracts.append(file_extract)
                except Exception as e:
                    # Create error extract for this specific file
                    error_extract = ExtractedContent(
                        content=f"Error processing {file_info.path}: {e}",
                        extraction_method="error",
                        metadata={
                            "error": str(e),
                            "source": str(file_info.path),
                            "requires_api_upload": False,
                        },
                        file_info=file_info,
                    )
                    file_extracts.append(error_extract)

            return file_extracts

        except Exception as e:
            # Return error for the entire directory
            return [
                ExtractedContent(
                    content=f"Error expanding directory {directory_path}: {e}",
                    extraction_method="error",
                    metadata={
                        "error": str(e),
                        "source": str(directory_path),
                        "requires_api_upload": False,
                    },
                    file_info=directory_extract.file_info,
                ),
            ]

    def create_api_parts(
        self,
        extracted_contents: list[ExtractedContent],
        cache_enabled: bool = False,  # noqa: FBT001, FBT002
        client: Any | None = None,
    ) -> list[types.Part]:
        """Coordinate conversion of extracted content to API parts.

        Args:
            extracted_contents: List of processed content
            cache_enabled: Whether caching is enabled (affects upload decisions)
            client: Gemini client for file uploads when needed

        Returns:
            List of Gemini API Part objects
        """
        log.info(
            "Creating API parts for %d content items (cache_enabled=%s)",
            len(extracted_contents),
            cache_enabled,
        )

        api_parts = []
        is_multimodal = any(
            self.file_ops.is_multimodal_content(e) for e in extracted_contents
        )

        for i, extracted in enumerate(extracted_contents):  # noqa: B007
            # Make upload decision based on strategy and caching
            strategy = self._determine_processing_strategy(
                extracted,
                cache_enabled,
                is_multimodal,
            )

            # Create appropriate API part
            if strategy == "text_only":
                part = types.Part(text=extracted.content)
            elif strategy == "url":
                part = self._create_url_part(extracted)
            elif strategy == "upload":
                part = self._create_upload_part(extracted, client)
            else:  # strategy == "inline" or fallback
                part = self._create_inline_part(extracted)

            api_parts.append(part)

        log.info("Successfully created %d API parts", len(api_parts))
        return api_parts

    def _determine_processing_strategy(
        self,
        extracted: ExtractedContent,
        cache_enabled: bool,  # noqa: FBT001
        is_multimodal: bool,  # noqa: FBT001
    ) -> str:
        """Determine processing strategy, forcing upload for multimodal if caching."""
        strategy = extracted.processing_strategy

        if cache_enabled and strategy == "inline" and is_multimodal:
            log.debug("Forcing upload strategy for multimodal content due to caching.")
            return "upload"

        log.debug(
            "Determined processing strategy '%s' for source '%s'",
            strategy,
            extracted.file_info.path,
        )
        return strategy

    def _create_upload_part(
        self,
        extracted: ExtractedContent,
        client: Any | None,
    ) -> types.Part:
        """Coordinate file upload through low-level utility"""  # noqa: D415
        # Handle arXiv PDFs that need download-first-then-upload
        if extracted.extraction_method == "arxiv_pdf_download":
            log.debug("Handling arXiv PDF upload for: %s", extracted.file_info.path)
            return self._handle_arxiv_pdf_upload(extracted, client)

        # Regular file upload path (existing logic unchanged)
        if not extracted.file_path:
            log.error(
                "Content requires upload but no file path available for: %s",
                extracted.file_info.path,
            )
            raise APIError("Content requires upload but no file path available")

        if client is None:
            log.error(
                "Client required for file upload but not provided for: %s",
                extracted.file_info.path,
            )
            raise APIError("Client required for file upload but not provided")

        log.debug("Uploading file: %s", extracted.file_path)
        upload_manager = self._get_file_upload_manager(client)
        uploaded_file = upload_manager.upload_and_wait(extracted.file_path)

        mime_type = extracted.metadata.get("mime_type") or getattr(
            uploaded_file,
            "mime_type",
            None,
        )
        return types.Part(
            file_data=types.FileData(file_uri=uploaded_file.uri, mime_type=mime_type),
        )

    def _handle_arxiv_pdf_upload(
        self,
        extracted: ExtractedContent,
        client: Any | None,
    ) -> types.Part:
        """Handle arXiv PDF upload using Files API (for large PDFs)"""  # noqa: D415
        if client is None:
            raise APIError("Client required for arXiv PDF upload but not provided")

        source_url = extracted.metadata.get("url")
        if not source_url:
            raise APIError("arXiv PDF extraction missing URL")

        # Download content to memory
        content_bytes = self._download_url_content(extracted, source_url)

        # Use Files API with BytesIO
        doc_io = io.BytesIO(content_bytes)

        uploaded_file = client.files.upload(
            file=doc_io,
            config={"mime_type": "application/pdf"},
        )

        # Return proper Part with file data (fixed: was returning raw uploaded_file)
        return types.Part(
            file_data=types.FileData(
                file_uri=uploaded_file.uri,
                mime_type="application/pdf",
            ),
        )

    def _create_url_part(self, extracted: ExtractedContent) -> types.Part:
        """Create part from URL-based content (YouTube, PDF URLs)"""  # noqa: D415
        url = extracted.metadata.get("url")
        if url:
            return types.Part(file_data=types.FileData(file_uri=url))
        # Fallback to text if URL is missing
        return types.Part(text=extracted.content)

    def _create_inline_part(self, extracted: ExtractedContent) -> types.Part:
        """Create inline part from extracted content - handles URLs and files"""  # noqa: D415
        if extracted.file_path:
            # File path - read file data
            content_bytes = extracted.file_path.read_bytes()
            mime_type = extracted.metadata.get("mime_type", "application/octet-stream")
            log.debug(
                "Read %d bytes from file: %s",
                len(content_bytes),
                extracted.file_path,
            )
        else:
            # Check if this is URL content that needs to be downloaded
            source_url = extracted.metadata.get("url")
            if source_url:
                # URL content - download it (restore the missing logic)
                content_bytes = self._download_url_content(extracted, source_url)
                mime_type = extracted.metadata.get(
                    "mime_type",
                    "application/octet-stream",
                )
                log.debug(
                    "Downloaded %d bytes from URL: %s",
                    len(content_bytes),
                    source_url,
                )
            else:
                # Text content - encode to bytes
                content_bytes = extracted.content.encode("utf-8")
                mime_type = "text/plain"
                log.debug("Encoded %d characters to bytes", len(extracted.content))

        return types.Part.from_bytes(
            data=content_bytes,
            mime_type=mime_type,
        )

    def _download_url_content(
        self,
        extracted: ExtractedContent,
        source_url: str,
    ) -> bytes:
        """Download URL content"""  # noqa: D415
        # Check if content is cached to avoid double download
        if extracted.metadata.get("content_cached", False):
            # Get cached content from URLExtractor
            url_extractor = None
            for extractor in self.file_ops.extractor_manager.extractors:
                if hasattr(extractor, "get_cached_content"):
                    url_extractor = extractor
                    break

            if url_extractor:
                file_data = url_extractor.get_cached_content(source_url)
                if file_data is None:
                    # Cache miss - download using extractor's method
                    file_data = url_extractor.download_content_if_needed(source_url)
            else:
                # Fallback to direct download
                with httpx.Client(timeout=30) as client:
                    response = client.get(source_url)
                    response.raise_for_status()
                    file_data = response.content
        else:
            # Content not cached - download it
            with httpx.Client(timeout=30) as client:
                response = client.get(source_url)
                response.raise_for_status()
                file_data = response.content

        return file_data

    def separate_cacheable_content(
        self,
        parts: list[types.Part],
    ) -> tuple[list[types.Part], list[types.Part]]:
        """Separate content parts from prompt parts for explicit caching"""  # noqa: D415
        # Heuristic: cache large text and all file parts. Keep small text as prompt.
        content_parts = []
        prompt_parts = []
        for part in parts:
            if self._is_large_text_part(part) or self._is_file_part(part):
                content_parts.append(part)
            else:
                prompt_parts.append(part)
        return content_parts, prompt_parts

    def optimize_for_implicit_cache(self, parts: list[types.Part]) -> list[types.Part]:
        """Prepares parts for implicit caching by merging text parts"""  # noqa: D415
        if not parts:
            return []

        # Find the first text part to merge subsequent text parts into
        first_text_index = -1
        for i, part in enumerate(parts):
            if self._is_text_part(part):
                first_text_index = i
                break

        if first_text_index == -1:
            return parts  # No text parts to merge

        merged_text = ""
        new_parts = list(parts[:first_text_index])

        for part in parts[first_text_index:]:
            if self._is_text_part(part):
                merged_text += part.text
            else:
                if merged_text:
                    new_parts.append(merged_text)
                    merged_text = ""
                new_parts.append(part)

        if merged_text:
            new_parts.append(merged_text)

        return new_parts

    def _is_text_part(self, part) -> bool:
        """Check if a part is a simple text part."""
        # In the native library, text parts can be strings
        return isinstance(part, str) or (
            hasattr(part, "text") and not hasattr(part, "file_data")
        )

    def _is_file_part(self, part) -> bool:
        """Check if a part is a file/blob part."""
        return hasattr(part, "file_data") or isinstance(part, types.Blob)

    def _is_large_text_part(self, part, threshold=LARGE_TEXT_THRESHOLD) -> bool:
        """Check if a text part is considered large."""
        if not self._is_text_part(part):
            return False

        text_content = part if isinstance(part, str) else part.text
        # A simple character count heuristic
        # A more accurate method would be to use a tokenizer
        return len(text_content) > threshold
