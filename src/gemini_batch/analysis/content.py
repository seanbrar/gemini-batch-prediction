from dataclasses import dataclass
from pathlib import Path

from ..client.content_processor import ContentProcessor
from ..exceptions import ValidationError
from ..files import FileType


@dataclass
class SourceSummary:
    """Source breakdown for efficiency demonstration"""

    total_count: int
    breakdown: dict[str, int]  # source_type -> count
    file_types: dict[FileType, int]  # file_type -> count
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
        "error": "error",
        "empty_directory": "empty_directory",
    }

    def __init__(self):
        self.content_processor = ContentProcessor()

    def analyze_sources(
        self,
        sources: str | Path | list[str | Path],
        questions: list[str] | None = None,
    ) -> SourceSummary:
        """Analyze sources for batch processing efficiency demonstration"""
        if not sources:
            raise ValidationError("Sources cannot be empty")

        # Use ContentProcessor to expand sources (handles directories)
        extracted_contents = self.content_processor.process_content(sources)

        # Convert to analysis format
        expanded_sources = self._convert_to_analysis_format(extracted_contents)

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

    def _convert_to_analysis_format(self, extracted_contents) -> list[dict[str, any]]:
        """Convert ExtractedContent list to analysis format"""
        expanded = []

        for extracted in extracted_contents:
            expanded.append(
                {
                    "source": extracted.file_info.path,
                    "type": self._classify_source_type(extracted),
                    "file_type": extracted.file_info.file_type,
                    "extraction_method": extracted.extraction_method,
                    "error": extracted.metadata.get("error")
                    if extracted.extraction_method == "error"
                    else None,
                },
            )

        return expanded

    def _classify_source_type(self, extracted_content) -> str:
        """Simple source type classification"""
        method = extracted_content.extraction_method
        return self.EXTRACTION_METHOD_TO_TYPE.get(method, "local_file")

    def _categorize_sources(self, expanded_sources: list[dict]) -> dict[str, int]:
        """Count sources by type for demo display"""
        breakdown = {}

        for source in expanded_sources:
            source_type = source["type"]
            breakdown[source_type] = breakdown.get(source_type, 0) + 1

        return breakdown

    def _categorize_file_types(
        self,
        expanded_sources: list[dict],
    ) -> dict[FileType, int]:
        """Count sources by file type"""
        file_types = {}

        for source in expanded_sources:
            file_type = source.get("file_type", FileType.UNKNOWN)
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return file_types
