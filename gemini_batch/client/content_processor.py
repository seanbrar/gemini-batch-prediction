"""
Content processing orchestration - converts sources to extracted content
"""

from pathlib import Path
from typing import List, Union

from ..files import FileOperations
from ..files.extractors import ExtractedContent
from ..files.scanner import FileInfo, FileType


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
    """Pure orchestration - converts sources to extracted content only"""

    def __init__(self):
        self.file_ops = FileOperations()

    def process_content(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[ExtractedContent]:
        """Convert any content source into list of ExtractedContent"""
        # Flatten nested lists first to avoid deep recursion
        flattened_sources = _flatten_sources(source)

        extracted_contents = []
        for single_source in flattened_sources:
            # Process each flattened source (might expand directories)
            source_extracts = self._process_single_source(single_source)
            extracted_contents.extend(source_extracts)

        return extracted_contents

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
            # Return error info in a structured way for the API client to handle
            # Create a minimal FileInfo for error cases
            source_path = Path(str(source))
            error_file_info = FileInfo(
                path=source_path,
                file_type=FileType.UNKNOWN,
                size=0,
                extension=source_path.suffix,
                name=source_path.name,
                relative_path=source_path,  # Required field
                mime_type="text/plain",
            )

            error_extract = ExtractedContent(
                content=f"Error processing {source}: {e}",
                extraction_method="error",
                metadata={"error": str(e), "source": str(source)},
                file_info=error_file_info,  # Required parameter
                requires_api_upload=False,
            )
            return [error_extract]

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
