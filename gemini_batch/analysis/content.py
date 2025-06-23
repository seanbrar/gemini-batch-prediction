from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..exceptions import ValidationError
from ..files import FileOperations, FileType


@dataclass
class SourceSummary:
    """Source breakdown for efficiency demonstration"""

    total_count: int
    breakdown: Dict[str, int]  # source_type -> count
    file_types: Dict[FileType, int]  # file_type -> count
    traditional_api_calls: int  # sources × questions
    batch_api_calls: int  # typically 1
    efficiency_factor: float  # traditional / batch


class ContentAnalyzer:
    """Source analysis for batch processing efficiency demonstration"""

    EXTRACTION_METHOD_TO_TYPE = {
        "direct_text": "text_content",
        "youtube_api": "youtube_video",
        "pdf_url_api": "arxiv_paper",
        "url_download": "web_content",
        "directory_scan": "file_directory",
    }

    def __init__(self):
        self.file_ops = FileOperations()

    def analyze_sources(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        questions: Optional[List[str]] = None,
    ) -> SourceSummary:
        """Analyze sources for batch processing efficiency demonstration"""
        if not sources:
            raise ValidationError("Sources cannot be empty")

        # Expand sources to get actual count (handles directories)
        expanded_sources = self._expand_sources(sources)

        # Categorize sources
        breakdown = self._categorize_sources(expanded_sources)
        file_types = self._categorize_file_types(expanded_sources)

        # Calculate efficiency
        question_count = len(questions) if questions else 1
        traditional_calls = len(expanded_sources) * question_count
        batch_calls = 1  # Single batch call
        efficiency_factor = traditional_calls / batch_calls

        return SourceSummary(
            total_count=len(expanded_sources),
            breakdown=breakdown,
            file_types=file_types,
            traditional_api_calls=traditional_calls,
            batch_api_calls=batch_calls,
            efficiency_factor=efficiency_factor,
        )

    def _expand_sources(self, sources) -> List[Dict[str, any]]:
        """Expand sources using existing file operations - focus on counting"""
        if not isinstance(sources, list):
            sources = [sources]

        expanded = []

        for source in sources:
            try:
                # Use existing file operations to handle directories, etc.
                extracted = self.file_ops.process_source(source)

                # If it's a directory, we need to count the actual files
                if extracted.extraction_method == "directory_scan":
                    # Get directory contents
                    directory_path = Path(extracted.metadata["directory_path"])
                    scan_results = self.file_ops.scan_directory(directory_path)

                    # Add each file as a separate source
                    for file_type, files in scan_results.items():
                        for file_info in files:
                            expanded.append(
                                {
                                    "source": file_info.path,
                                    "type": "local_file",
                                    "file_type": file_type,
                                    "from_directory": str(directory_path),
                                }
                            )
                else:
                    # Single source
                    expanded.append(
                        {
                            "source": source,
                            "type": self._classify_source_type(extracted),
                            "file_type": extracted.file_info.file_type
                            if extracted.file_info
                            else FileType.UNKNOWN,
                            "extraction_method": extracted.extraction_method,
                        }
                    )

            except Exception as e:
                # For demo purposes, we want to be lenient but informative
                expanded.append(
                    {
                        "source": source,
                        "type": "unknown",
                        "file_type": FileType.UNKNOWN,
                        "error": str(e),
                    }
                )

        return expanded

    def _classify_source_type(self, extracted_content) -> str:
        """Simple source type classification"""
        method = extracted_content.extraction_method
        return self.EXTRACTION_METHOD_TO_TYPE.get(method, "local_file")

    def _categorize_sources(self, expanded_sources: List[Dict]) -> Dict[str, int]:
        """Count sources by type for demo display"""
        breakdown = {}

        for source in expanded_sources:
            source_type = source["type"]
            breakdown[source_type] = breakdown.get(source_type, 0) + 1

        return breakdown

    def _categorize_file_types(
        self, expanded_sources: List[Dict]
    ) -> Dict[FileType, int]:
        """Count sources by file type"""
        file_types = {}

        for source in expanded_sources:
            file_type = source.get("file_type", FileType.UNKNOWN)
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return file_types
