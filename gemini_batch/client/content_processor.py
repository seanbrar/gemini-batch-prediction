"""
Content processing orchestration - converts multiple mixed sources
into processed ExtractedContent objects for API consumption
"""

from pathlib import Path
from typing import Any, List, Optional, Union

from google.genai import types

from ..exceptions import APIError
from ..files import FileOperations
from ..files.extractors import ExtractedContent
from .file_upload_manager import FileUploadManager


def _flatten_sources(sources: Union[str, Path, List]) -> List[Union[str, Path]]:
    """Flatten nested source lists to avoid deep recursion in processing"""
    if not isinstance(sources, list):
        return [sources]

    flattened = []
    for item in sources:
        if isinstance(item, list):
            flattened.extend(_flatten_sources(item))
        else:
            flattened.append(item)
    return flattened


class ContentProcessor:
    """
    Content orchestration layer - handles multiple mixed sources and batch processing.

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

    def __init__(self):
        self.file_ops = FileOperations()
        # Lazy initialization of file upload manager
        self._file_upload_manager: Optional[FileUploadManager] = None

    def _get_file_upload_manager(self, client: Any) -> FileUploadManager:
        """Lazy initialization of file upload manager"""
        if self._file_upload_manager is None:
            self._file_upload_manager = FileUploadManager(client)
        return self._file_upload_manager

    def process_content(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[ExtractedContent]:
        """Convert any content source into list of ExtractedContent for API consumption"""
        # Flatten nested lists first to avoid deep recursion
        flattened_sources = _flatten_sources(source)

        extracted_contents = []
        for single_source in flattened_sources:
            # Process each flattened source (might expand directories)
            source_extracts = self._process_single_source(single_source)
            extracted_contents.extend(source_extracts)

        return extracted_contents

    def is_multimodal_content(self, extracted_content: ExtractedContent) -> bool:
        """Check if content is multimodal by delegating to FileOperations"""
        return self.file_ops.is_multimodal_content(extracted_content)

    def _process_single_source(
        self, source: Union[str, Path]
    ) -> List[ExtractedContent]:
        """Process a single content source to list of ExtractedContent (directories expand to multiple)"""
        try:
            # Get initial extraction from file operations
            extracted_content = self.file_ops.process_source(source)

            # Check if this is a directory marker that needs expansion
            if extracted_content.extraction_method == "directory_scan":
                return self._expand_directory(extracted_content)
            else:
                return [extracted_content]

        except Exception as e:
            # Let the error bubble up rather than trying to create invalid ExtractedContent
            # The caller (GeminiClient) can handle this more appropriately
            raise RuntimeError(f"Error processing {source}: {e}") from e

    def _expand_directory(
        self, directory_extract: ExtractedContent
    ) -> List[ExtractedContent]:
        """Expand a directory marker into individual file extractions"""
        directory_path = Path(directory_extract.metadata["directory_path"])

        try:
            # Use file operations to scan the directory
            categorized_files = self.file_ops.scan_directory(directory_path)

            # Flatten all files into a single list
            all_files = []
            for file_type, files in categorized_files.items():
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
                        },
                        file_info=directory_extract.file_info,
                        requires_api_upload=False,
                    )
                ]

            # Process each file individually
            file_extracts = []
            for file_info in all_files:
                try:
                    file_extract = self.file_ops.extract_content(file_info)
                    file_extracts.append(file_extract)
                except Exception as e:
                    # Create error extract for this specific file
                    error_extract = ExtractedContent(
                        content=f"Error processing {file_info.path}: {e}",
                        extraction_method="error",
                        metadata={"error": str(e), "source": str(file_info.path)},
                        file_info=file_info,
                        requires_api_upload=False,
                    )
                    file_extracts.append(error_extract)

            return file_extracts

        except Exception as e:
            # Return error for the entire directory
            return [
                ExtractedContent(
                    content=f"Error expanding directory {directory_path}: {e}",
                    extraction_method="error",
                    metadata={"error": str(e), "source": str(directory_path)},
                    file_info=directory_extract.file_info,
                    requires_api_upload=False,
                )
            ]

    def create_api_parts(
        self,
        extracted_contents: List[ExtractedContent],
        cache_enabled: bool = False,
        client: Optional[Any] = None,
    ) -> List[types.Part]:
        """
        Coordinate conversion of extracted content to API parts.

        High-level components call this and don't need to know about:
        - Upload decisions
        - Multimodal detection
        - Caching implications
        - File upload coordination

        Args:
            extracted_contents: List of processed content
            cache_enabled: Whether caching is enabled (affects upload decisions)
            client: Gemini client for file uploads when needed

        Returns:
            List of Gemini API Part objects
        """
        api_parts = []

        for extracted in extracted_contents:
            # Make upload decision based on strategy and caching
            strategy = self._determine_processing_strategy(extracted, cache_enabled)

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

        return api_parts

    def _determine_processing_strategy(
        self, extracted: ExtractedContent, cache_enabled: bool
    ) -> str:
        """
        Coordinate processing strategy decisions.
        Uses existing file_ops for multimodal detection.
        """
        base_strategy = extracted.processing_strategy

        # If caching enabled and multimodal content would be inline, force upload
        if (
            cache_enabled
            and base_strategy == "inline"
            and self.file_ops.is_multimodal_content(extracted)
        ):
            return "upload"

        return base_strategy

    def _create_upload_part(
        self, extracted: ExtractedContent, client: Optional[Any]
    ) -> types.Part:
        """Coordinate file upload through low-level utility"""
        if not extracted.file_path:
            raise APIError("Content requires upload but no file path available")

        if client is None:
            raise APIError("Client required for file upload but not provided")

        # Coordinate with low-level file upload manager
        upload_manager = self._get_file_upload_manager(client)
        uploaded_file = upload_manager.upload_and_wait(extracted.file_path)

        # Build API part with mime_type for cache compatibility
        mime_type = extracted.metadata.get("mime_type") or getattr(
            uploaded_file, "mime_type", None
        )
        return types.Part(
            file_data=types.FileData(file_uri=uploaded_file.uri, mime_type=mime_type)
        )

    def _create_url_part(self, extracted: ExtractedContent) -> types.Part:
        """Create part from URL-based content (YouTube, PDF URLs)"""
        url = extracted.metadata.get("url")
        if url:
            return types.Part(file_data=types.FileData(file_uri=url))
        else:
            # Fallback to text if URL is missing
            return types.Part(text=extracted.content)

    def _create_inline_part(self, extracted: ExtractedContent) -> types.Part:
        """Create inline part from extracted content"""
        if extracted.file_path:
            # File path - read file data
            content_bytes = extracted.file_path.read_bytes()
        else:
            # Convert string content to bytes
            content_bytes = extracted.content.encode("utf-8")

        return types.Part.from_bytes(
            data=content_bytes,
            mime_type=extracted.metadata.get("mime_type", "text/plain"),
        )
