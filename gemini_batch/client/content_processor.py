"""
Content processing orchestration - converts sources to extracted content
"""

from pathlib import Path
from typing import List, Union

from ..files import FileOperations
from ..files.extractors import ExtractedContent
from ..files.scanner import FileInfo, FileType


class ContentProcessor:
    """Pure orchestration - converts sources to extracted content only"""

    def __init__(self, file_ops: FileOperations):
        self.file_ops = file_ops

    def process_content(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[ExtractedContent]:
        """Convert any content source into list of ExtractedContent"""
        if isinstance(source, list):
            extracted_contents = []
            for item in source:
                extracted_contents.append(self._process_single_source(item))
            return extracted_contents
        else:
            return [self._process_single_source(source)]

    def _process_single_source(self, source: Union[str, Path]) -> ExtractedContent:
        """Process a single content source to ExtractedContent"""
        try:
            # Pure file processing - no API parts creation
            return self.file_ops.process_source(source)
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

            return ExtractedContent(
                content=f"Error processing {source}: {e}",
                extraction_method="error",
                metadata={"error": str(e), "source": str(source)},
                file_info=error_file_info,  # Required parameter
                requires_api_upload=False,
            )
