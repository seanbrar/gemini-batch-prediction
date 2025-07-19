from dataclasses import dataclass  # noqa: D100
from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: UP035

from ..client.content_processor import ContentProcessor  # noqa: TID252
from ..exceptions import ValidationError  # noqa: TID252
from ..files import FileType  # noqa: TID252


@dataclass
class SourceSummary:
    """Source breakdown for efficiency demonstration"""  # noqa: D415

    total_count: int
    breakdown: Dict[str, int]  # source_type -> count  # noqa: UP006
    file_types: Dict[FileType, int]  # file_type -> count  # noqa: UP006
    traditional_api_calls: int  # sources × questions  # noqa: RUF003
    batch_api_calls: int  # typically 1
    efficiency_factor: float  # traditional / batch


class ContentAnalyzer:
    """Source analysis for batch processing efficiency demonstration"""  # noqa: D415

    EXTRACTION_METHOD_TO_TYPE = {  # noqa: RUF012
        "direct_text": "text_content",
        "youtube_api": "youtube_video",
        "pdf_url_api": "arxiv_paper",
        "url_download": "web_content",
        "directory_scan": "file_directory",
        "error": "error",
        "empty_directory": "empty_directory",
    }

    def __init__(self):  # noqa: ANN204, D107
        self.content_processor = ContentProcessor()

    def analyze_sources(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],  # noqa: UP006, UP007
        questions: Optional[List[str]] = None,  # noqa: UP006, UP045
    ) -> SourceSummary:
        """Analyze sources for batch processing efficiency demonstration"""  # noqa: D415
        if not sources:
            raise ValidationError("Sources cannot be empty")  # noqa: EM101, TRY003

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

    def _convert_to_analysis_format(self, extracted_contents) -> List[Dict[str, any]]:  # noqa: ANN001, UP006
        """Convert ExtractedContent list to analysis format"""  # noqa: D415
        expanded = []

        for extracted in extracted_contents:
            expanded.append(  # noqa: PERF401
                {
                    "source": extracted.file_info.path,
                    "type": self._classify_source_type(extracted),
                    "file_type": extracted.file_info.file_type,
                    "extraction_method": extracted.extraction_method,
                    "error": extracted.metadata.get("error")
                    if extracted.extraction_method == "error"
                    else None,
                }  # noqa: COM812
            )

        return expanded

    def _classify_source_type(self, extracted_content) -> str:  # noqa: ANN001
        """Simple source type classification"""  # noqa: D415
        method = extracted_content.extraction_method
        return self.EXTRACTION_METHOD_TO_TYPE.get(method, "local_file")

    def _categorize_sources(self, expanded_sources: List[Dict]) -> Dict[str, int]:  # noqa: UP006
        """Count sources by type for demo display"""  # noqa: D415
        breakdown = {}

        for source in expanded_sources:
            source_type = source["type"]
            breakdown[source_type] = breakdown.get(source_type, 0) + 1

        return breakdown

    def _categorize_file_types(
        self, expanded_sources: List[Dict]  # noqa: COM812, UP006
    ) -> Dict[FileType, int]:  # noqa: UP006
        """Count sources by file type"""  # noqa: D415
        file_types = {}

        for source in expanded_sources:
            file_type = source.get("file_type", FileType.UNKNOWN)
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return file_types
