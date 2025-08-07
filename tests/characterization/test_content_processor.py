"""
Characterization tests for the ContentProcessor.

These tests verify the behavior of the library's input pipeline, ensuring
that it correctly processes various source types like files, directories,
and URLs. We use `pyfakefs` to create a virtual filesystem for file-based
tests and mock `httpx` for URL-based tests.
"""

from unittest.mock import MagicMock

import pytest


def _serialize_extracted_content(extracted_contents):
    """
    Helper function to convert a list of ExtractedContent objects into a
    stable, serializable format for golden file comparison.

    Note: This function is kept for backward compatibility but the new architecture
    handles content processing differently through the SourceHandler.
    """
    if not isinstance(extracted_contents, list):
        extracted_contents = [extracted_contents]

    serializable_output = []
    for item in extracted_contents:
        serializable_output.append(
            {
                "content": item.content,
                "extraction_method": item.extraction_method,
                "file_info": {
                    "name": item.file_info.name,
                    "file_type": item.file_info.file_type.value,
                    "extension": item.file_info.extension,
                    "mime_type": item.file_info.mime_type,
                },
                "requires_api_upload": item.requires_api_upload,
                "processing_strategy": item.processing_strategy,
            }
        )
    return serializable_output


@pytest.mark.golden_test("golden_files/test_content_processor_directory.yml")
def test_directory_processing_behavior(golden, fs, mock_get_mime_type, batch_processor):  # noqa: ARG001
    """
    Characterizes the behavior of processing a directory source through the new architecture.
    """
    # Arrange
    fake_directory_path = "/test_data/docs"
    fs.create_dir(fake_directory_path)
    fs.create_file(
        f"{fake_directory_path}/main_document.txt",
        contents="This is the primary text file.",
    )
    fs.create_file(
        f"{fake_directory_path}/notes.md", contents="# These are some notes."
    )
    fs.create_file(f"{fake_directory_path}/.DS_Store", contents="")

    source_path = golden["input"]["source_path"]
    questions = ["What is the content of these documents?"]

    # Act - Test through the batch processor interface
    # This will exercise the new SourceHandler internally
    result = batch_processor.process_questions(source_path, questions)

    # For now, we'll just verify the processor doesn't crash
    # The actual content processing behavior will be tested once SourceHandler is implemented
    assert isinstance(result, dict)
    # TODO: Once SourceHandler is implemented, we can extract and verify the resolved sources


@pytest.mark.golden_test("golden_files/test_content_processor_youtube_url.yml")
def test_url_youtube_processing_behavior(golden, batch_processor):
    """
    Characterizes how a YouTube URL is processed through the new architecture.
    """
    # Arrange
    source_url = golden["input"]["source_url"]
    questions = ["What is this video about?"]

    # Act - Test through the batch processor interface
    result = batch_processor.process_questions(source_url, questions)

    # For now, we'll just verify the processor doesn't crash
    assert isinstance(result, dict)
    # TODO: Once SourceHandler is implemented, we can extract and verify the resolved sources


@pytest.mark.golden_test("golden_files/test_content_processor_arxiv_url.yml")
def test_url_arxiv_processing_behavior(golden, mock_httpx_client, batch_processor):
    """
    Characterizes how an arXiv URL is processed through the new architecture.
    """
    # Arrange
    source_url = golden["input"]["source_url"]
    questions = ["What is this paper about?"]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {
        "content-type": "application/pdf",
        "content-length": "123456",
    }
    mock_response.content = b"%PDF-1.4 fake content"
    mock_httpx_client.get.return_value = mock_response
    mock_httpx_client.head.return_value = mock_response

    # Act - Test through the batch processor interface
    result = batch_processor.process_questions(source_url, questions)

    # For now, we'll just verify the processor doesn't crash
    assert isinstance(result, dict)
    # TODO: Once SourceHandler is implemented, we can extract and verify the resolved sources


@pytest.mark.golden_test("golden_files/test_content_processor_local_image.yml")
def test_local_multimodal_file_behavior(
    golden, fs, _mock_get_mime_type, batch_processor
):
    """
    Characterizes processing of a local, non-text file (an image) through the new architecture.
    """
    # Arrange
    fake_image_path = "/test_data/image.png"
    fs.create_file(fake_image_path, contents=b"fakepngcontent")
    source_path = golden["input"]["source_path"]
    questions = ["What is in this image?"]

    # Act - Test through the batch processor interface
    result = batch_processor.process_questions(source_path, questions)

    # For now, we'll just verify the processor doesn't crash
    assert isinstance(result, dict)
    # TODO: Once SourceHandler is implemented, we can extract and verify the resolved sources


@pytest.mark.golden_test("golden_files/test_content_processor_mixed_list.yml")
def test_mixed_content_list_behavior(golden, fs, _mock_get_mime_type, batch_processor):
    """
    Characterizes processing of a list containing mixed content types through the new architecture.
    """
    # Arrange
    fs.create_file("/test_data/report.txt", contents="Text from a file.")
    mixed_source_list = golden["input"]["source_list"]
    questions = ["What is the content across all these sources?"]

    # Act - Test through the batch processor interface
    result = batch_processor.process_questions(mixed_source_list, questions)

    # For now, we'll just verify the processor doesn't crash
    assert isinstance(result, dict)
    # TODO: Once SourceHandler is implemented, we can extract and verify the resolved sources
