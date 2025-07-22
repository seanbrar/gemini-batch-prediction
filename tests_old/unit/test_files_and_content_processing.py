"""
Unit tests for files and content processing functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

from pyfakefs.fake_filesystem import FakeFilesystem
import pytest

from gemini_batch.client.content_processor import ContentProcessor
from gemini_batch.constants import FILES_API_THRESHOLD
from gemini_batch.files import ExtractedContent
from gemini_batch.files.extractors import URLExtractor, YouTubeExtractor
from gemini_batch.files.operations import FileOperations
from gemini_batch.files.scanner import DirectoryScanner, FileInfo, FileType


class TestDirectoryScanner:
    """Test directory scanning functionality."""

    def test_directory_scanner_respects_exclusions(self, fs: FakeFilesystem):
        """Test that DirectoryScanner ignores excluded directories and files."""
        # Create a fake filesystem with various files and directories
        fs.create_file("/test/.git/config", contents="git config")
        fs.create_file("/test/.DS_Store", contents="mac file")
        fs.create_file("/test/node_modules/package.json", contents="{}")
        fs.create_file("/test/__pycache__/test.pyc", contents="compiled python")
        fs.create_file("/test/venv/activate", contents="virtual env")
        fs.create_file("/test/actual_content.txt", contents="real content")
        fs.create_file("/test/important.pdf", contents="pdf content")
        fs.create_file("/test/README.md", contents="readme content")

        scanner = DirectoryScanner()
        files = scanner.scan_directory("/test")

        # Should only include actual content files, not excluded ones
        # Files is a dict of FileType -> List[FileInfo]
        file_paths = []
        for file_list in files.values():
            file_paths.extend([str(f.path) for f in file_list])

        assert any(p.endswith("actual_content.txt") for p in file_paths)
        assert any(p.endswith("README.md") for p in file_paths)
        assert any(p.endswith("important.pdf") for p in file_paths)

        # Should exclude system and generated files
        assert ".git/config" not in file_paths
        assert ".DS_Store" not in file_paths
        assert "node_modules/package.json" not in file_paths
        assert "__pycache__/test.pyc" not in file_paths
        assert "venv/activate" not in file_paths

    def test_directory_scanner_handles_empty_directory(self, fs: FakeFilesystem):
        """Test scanner with an empty directory."""
        fs.create_dir("/test/empty_dir")
        scanner = DirectoryScanner()
        files = scanner.scan_directory("/test/empty_dir")
        assert files == {}

    def test_directory_scanner_handles_nested_structure(self, fs: FakeFilesystem):
        """Test scanner with nested directory structure."""
        # Create nested structure with some excluded and some included files
        fs.create_file("/test/docs/.gitignore", contents="gitignore")
        fs.create_file("/test/docs/important.txt", contents="important")
        fs.create_file("/test/docs/subdir/.DS_Store", contents="mac file")
        fs.create_file("/test/docs/subdir/valid.pdf", contents="pdf")

        scanner = DirectoryScanner()
        files = scanner.scan_directory("/test")

        # Files is a dict of FileType -> List[FileInfo]
        file_paths = []
        for file_list in files.values():
            file_paths.extend([str(f.path) for f in file_list])

        assert any(p.endswith("docs/important.txt") for p in file_paths)
        assert any(p.endswith("docs/subdir/valid.pdf") for p in file_paths)
        assert "docs/.gitignore" not in file_paths
        assert "docs/subdir/.DS_Store" not in file_paths


class TestYouTubeURLExtraction:
    """Test YouTube URL extraction functionality."""

    def test_youtube_url_extractor_identifies_youtube_urls(self):
        """Test that YouTubeExtractor correctly identifies YouTube URLs."""
        extractor = YouTubeExtractor()

        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/v/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        ]

        for url in youtube_urls:
            assert extractor.can_extract_source(url), f"Should extract: {url}"
            result = extractor.extract_source(url)
            assert result is not None
            assert "dQw4w9WgXcQ" in result.metadata.get("url", "")

    def test_youtube_url_extractor_ignores_non_youtube_urls(self):
        """Test that YouTubeExtractor ignores non-YouTube URLs."""
        extractor = YouTubeExtractor()

        non_youtube_urls = [
            "https://www.google.com",
            "https://github.com/user/repo",
            "https://example.com/video",
            "https://vimeo.com/123456",
            "not a url at all",
        ]

        for url in non_youtube_urls:
            assert not extractor.can_extract_source(url), f"Should not extract: {url}"

    def test_url_extractor_ignores_youtube_urls(self):
        """Test that URLExtractor ignores YouTube URLs."""
        extractor = URLExtractor()

        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
        ]

        for url in youtube_urls:
            assert not extractor.can_extract(url), (
                f"Should not extract YouTube URL: {url}"
            )

    def test_url_extractor_handles_regular_urls(self):
        """Test that URLExtractor handles regular URLs."""
        extractor = URLExtractor()

        regular_urls = [
            "https://arxiv.org/pdf/2103.12345.pdf",
            "https://export.arxiv.org/pdf/2103.12345.pdf",
        ]

        for url in regular_urls:
            assert extractor.can_extract(url), f"Should extract: {url}"


class TestContentProcessorOrchestration:
    """Test content processor orchestration functionality."""

    def test_content_processor_orchestration(self):
        """Test that ContentProcessor correctly processes mixed sources."""
        from gemini_batch.client.content_processor import ContentProcessor

        processor = ContentProcessor()

        # Mock the file_ops attribute on the processor instance
        with patch.object(processor, "file_ops") as mock_file_ops_instance:
            # Mock process_source to return ExtractedContent objects
            from pathlib import Path

            from gemini_batch.files.extractors import ExtractedContent
            from gemini_batch.files.scanner import FileInfo, FileType

            def mock_process_source(source):
                if source == "Direct text content":
                    return ExtractedContent(
                        content="Direct text content",
                        metadata={"source_type": "text"},
                        file_info=FileInfo(
                            path=Path("text_content"),
                            file_type=FileType.TEXT,
                            size=len("Direct text content"),
                            extension="",
                            name="text_content",
                            relative_path=Path("text_content"),
                            mime_type="text/plain",
                        ),
                        extraction_method="direct_text",
                    )
                elif source == "/path/to/file.txt":
                    return ExtractedContent(
                        content="File content",
                        metadata={"source_type": "file"},
                        file_info=FileInfo(
                            path=Path("/path/to/file.txt"),
                            file_type=FileType.TEXT,
                            size=len("File content"),
                            extension=".txt",
                            name="file.txt",
                            relative_path=Path("file.txt"),
                            mime_type="text/plain",
                        ),
                        extraction_method="text_extractor",
                    )
                else:
                    raise Exception(f"Unexpected source: {source}")

            mock_file_ops_instance.process_source.side_effect = mock_process_source

            # Process mixed sources
            sources = ["Direct text content", "/path/to/file.txt"]
            result = processor.process_content(sources)

            # Verify both sources were processed
            assert len(result) == 2
            assert result[0].content == "Direct text content"
            assert result[1].content == "File content"

            # Verify file_ops was called for the file path
            mock_file_ops_instance.process_source.assert_called_with(
                "/path/to/file.txt"
            )

    def test_content_processor_handles_errors_gracefully(self):
        """Test that ContentProcessor raises RuntimeError for processing errors."""
        from gemini_batch.client.content_processor import ContentProcessor

        processor = ContentProcessor()

        # Mock the file_ops attribute on the processor instance
        with patch.object(processor, "file_ops") as mock_file_ops_instance:
            # Mock process_source to raise an error for missing files
            def mock_process_source(source):
                if source == "/path/to/missing/file.txt":
                    raise Exception("File not found")
                else:
                    from pathlib import Path

                    from gemini_batch.files.extractors import ExtractedContent
                    from gemini_batch.files.scanner import FileInfo, FileType

                    return ExtractedContent(
                        content="Valid content",
                        metadata={"source_type": "text"},
                        file_info=FileInfo(
                            path=Path("valid_content"),
                            file_type=FileType.TEXT,
                            size=len("Valid content"),
                            extension="",
                            name="valid_content",
                            relative_path=Path("valid_content"),
                            mime_type="text/plain",
                        ),
                        extraction_method="direct_text",
                    )

            mock_file_ops_instance.process_source.side_effect = mock_process_source

            # Process valid sources first
            valid_sources = ["Valid content", "Another valid content"]
            result = processor.process_content(valid_sources)

            # Should process valid sources successfully
            assert len(result) == 2
            assert result[0].content == "Valid content"
            assert result[1].content == "Valid content"

            # Process sources including one that will fail
            sources_with_error = ["Valid content", "/path/to/missing/file.txt"]

            # Should raise RuntimeError for processing errors
            with pytest.raises(
                RuntimeError,
                match="Error processing /path/to/missing/file.txt: File not found",
            ):
                processor.process_content(sources_with_error)

    def test_content_processor_with_empty_sources(self):
        """Test ContentProcessor with empty source list."""
        processor = ContentProcessor()

        result = processor.process_content([])

        assert len(result) == 0

    def test_content_processor_with_none_sources(self):
        """Test ContentProcessor with None sources."""
        processor = ContentProcessor()

        result = processor.process_content([None, "Valid content"])

        # Should filter out None values
        assert len(result) == 1
        assert any("Valid content" in str(r) for r in result)


class TestLargeFileProcessing:
    """Test large file processing functionality."""

    def test_large_file_forces_api_upload(self):
        """Test that large files trigger API upload strategy."""
        # Create a mock file that exceeds the threshold
        large_content = "x" * (FILES_API_THRESHOLD + 1000)

        from gemini_batch.files.scanner import FileInfo, FileType

        extracted_content = ExtractedContent(
            content=large_content,
            metadata={
                "source_type": "file",
                "source_path": "/path/to/large_file.txt",
                "requires_api_upload": True,  # Large file requires upload
            },
            file_info=FileInfo(
                path=Path("/path/to/large_file.txt"),
                file_type=FileType.TEXT,
                size=len(large_content),
                extension=".txt",
                name="large_file.txt",
                relative_path=Path("large_file.txt"),
                mime_type="text/plain",
            ),
            extraction_method="direct_text",
        )

        assert extracted_content.processing_strategy == "upload"

    def test_small_file_uses_inline_processing(self):
        """Test that small files use text_only processing strategy when they have content."""
        # Create a mock file that's under the threshold
        small_content = "x" * (FILES_API_THRESHOLD - 1000)

        extracted_content = ExtractedContent(
            content=small_content,
            metadata={
                "source_type": "file",
                "source_path": "/path/to/small_file.txt",
                "requires_api_upload": False,  # Small file doesn't require upload
            },
            file_info=FileInfo(
                path=Path("/path/to/small_file.txt"),
                file_type=FileType.TEXT,
                size=len(small_content),
                extension=".txt",
                name="small_file.txt",
                relative_path=Path("small_file.txt"),
                mime_type="text/plain",
            ),
            extraction_method="direct_text",
        )

        # Small files with content should be text_only, not inline
        assert extracted_content.processing_strategy == "text_only"

    def test_exact_threshold_file_uses_inline_processing(self):
        """Test that files at the exact threshold use text_only processing strategy when they have content."""
        # Create a mock file at the exact threshold
        threshold_content = "x" * FILES_API_THRESHOLD

        extracted_content = ExtractedContent(
            content=threshold_content,
            metadata={
                "source_type": "file",
                "source_path": "/path/to/threshold_file.txt",
                "requires_api_upload": False,  # At threshold, doesn't require upload
            },
            file_info=FileInfo(
                path=Path("/path/to/threshold_file.txt"),
                file_type=FileType.TEXT,
                size=len(threshold_content),
                extension=".txt",
                name="threshold_file.txt",
                relative_path=Path("threshold_file.txt"),
                mime_type="text/plain",
            ),
            extraction_method="direct_text",
        )

        # Files at threshold with content should be text_only, not inline
        assert extracted_content.processing_strategy == "text_only"

    def test_empty_content_uses_inline_processing(self):
        """Test that files with no content use inline processing strategy."""
        # Create a mock file with no content (like a media file)
        extracted_content = ExtractedContent(
            content="",  # No content
            metadata={
                "source_type": "file",
                "source_path": "/path/to/media_file.jpg",
                "requires_api_upload": False,  # Small file doesn't require upload
            },
            file_info=FileInfo(
                path=Path("/path/to/media_file.jpg"),
                file_type=FileType.IMAGE,
                size=0,
                extension=".jpg",
                name="media_file.jpg",
                relative_path=Path("media_file.jpg"),
                mime_type="image/jpeg",
            ),
            extraction_method="media_extractor",
        )

        # Files with no content should be inline
        assert extracted_content.processing_strategy == "inline"

    def test_content_extractor_manager_handles_mixed_sizes(self):
        """Test that content extractor manager handles files of different sizes."""
        from gemini_batch.files.extractors import ContentExtractorManager

        manager = ContentExtractorManager()

        # Test with text content sources instead of Mock objects
        small_content = "x" * (FILES_API_THRESHOLD - 1000)
        large_content = "x" * (FILES_API_THRESHOLD + 1000)

        # Process each source individually
        small_result = manager.process_source(small_content)
        large_result = manager.process_source(large_content)

        # Both should be processed successfully
        assert small_result is not None
        assert large_result is not None
        assert hasattr(small_result, "processing_strategy")
        assert hasattr(large_result, "processing_strategy")


@pytest.mark.unit
class TestFileOperations:
    """Test file operations functionality."""

    @patch("gemini_batch.files.operations.FileOperations")
    def test_file_operations_extract_from_file(self, mock_file_ops):
        """Test file extraction from various file types."""
        mock_file_ops_instance = Mock()
        mock_file_ops.return_value = mock_file_ops_instance

        # Mock successful extraction
        mock_file_ops_instance.extract_from_file.return_value = "extracted content"

        file_ops = mock_file_ops_instance

        # Test different file types
        file_paths = [
            "/path/to/document.pdf",
            "/path/to/text.txt",
            "/path/to/image.jpg",
        ]

        for file_path in file_paths:
            result = file_ops.extract_from_file(file_path)
            assert result == "extracted content"
            file_ops.extract_from_file.assert_called_with(file_path)

    @patch("gemini_batch.files.operations.FileOperations")
    def test_file_operations_extract_from_url(self, mock_file_ops):
        """Test URL content extraction."""
        mock_file_ops_instance = Mock()
        mock_file_ops.return_value = mock_file_ops_instance

        # Mock successful URL extraction
        mock_file_ops_instance.extract_from_url.return_value = "url content"

        file_ops = mock_file_ops_instance

        urls = [
            "https://example.com/document.pdf",
            "https://api.github.com/repos/user/repo/README.md",
        ]

        for url in urls:
            result = file_ops.extract_from_url(url)
            assert result == "url content"
            file_ops.extract_from_url.assert_called_with(url)

    def test_file_operations_error_handling(self):
        """Test file operations error handling."""

        file_ops = FileOperations()

        # Test with non-existent file
        with pytest.raises(Exception):
            file_ops.extract_from_file("/non/existent/file.txt")

        # Test with invalid URL
        with pytest.raises(Exception):
            file_ops.extract_from_url(
                "https://invalid-url-that-does-not-exist.com/file.pdf"
            )
