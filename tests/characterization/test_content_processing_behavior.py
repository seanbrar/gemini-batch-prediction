"""
Characterization tests for the ContentProcessor.

These tests verify the behavior of the library's input pipeline, ensuring
that it correctly processes various source types like files, directories,
and URLs. We use `pyfakefs` to create a virtual filesystem for file-based
tests and mock `httpx` for URL-based tests.
"""

from unittest.mock import MagicMock

import pytest

from gemini_batch.client.content_processor import ContentProcessor


def _serialize_extracted_content(extracted_contents):  # noqa: ANN001, ANN202
    """
    Helper function to convert a list of ExtractedContent objects into a
    stable, serializable format for golden file comparison.
    """
    if not isinstance(extracted_contents, list):
        extracted_contents = [extracted_contents]

    serializable_output = []
    for item in extracted_contents:
        serializable_output.append({  # noqa: PERF401
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
        })
    return serializable_output

@pytest.mark.golden_test("golden_files/test_content_processor_directory.yml")
def test_directory_processing_behavior(golden, fs, mock_get_mime_type):  # noqa: ANN001, ANN201, ARG001
    """
    Characterizes the behavior of processing a directory source.
    """
    # Arrange
    fake_directory_path = "/test_data/docs"
    fs.create_dir(fake_directory_path)
    fs.create_file(f"{fake_directory_path}/main_document.txt", contents="This is the primary text file.")  # noqa: E501
    fs.create_file(f"{fake_directory_path}/notes.md", contents="# These are some notes.")  # noqa: E501
    fs.create_file(f"{fake_directory_path}/.DS_Store", contents="")

    source_path = golden['input']['source_path']  # noqa: Q000
    content_processor = ContentProcessor()

    # Act
    extracted_contents = content_processor.process_content(source_path)
    serializable_output = _serialize_extracted_content(extracted_contents)

    # Assert
    assert serializable_output == golden.out['extracted_contents']  # noqa: Q000

@pytest.mark.golden_test("golden_files/test_content_processor_youtube_url.yml")
def test_url_youtube_processing_behavior(golden):  # noqa: ANN001, ANN201
    """
    Characterizes how a YouTube URL is processed.
    """
    # Arrange
    source_url = golden['input']['source_url']  # noqa: Q000
    content_processor = ContentProcessor()

    # Act
    extracted_content = content_processor.process_content(source_url)
    serializable_output = _serialize_extracted_content(extracted_content)

    # Assert
    assert serializable_output == golden.out['extracted_contents']  # noqa: Q000

@pytest.mark.golden_test("golden_files/test_content_processor_arxiv_url.yml")
def test_url_arxiv_processing_behavior(golden, mock_httpx_client):  # noqa: ANN001, ANN201
    """
    Characterizes how an arXiv URL is processed, mocking the network call.
    """
    # Arrange
    source_url = golden['input']['source_url']  # noqa: Q000
    content_processor = ContentProcessor()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/pdf', 'content-length': '123456'}  # noqa: E501, Q000
    mock_response.content = b'%PDF-1.4 fake content'  # noqa: Q000
    mock_httpx_client.get.return_value = mock_response
    mock_httpx_client.head.return_value = mock_response

    # Act
    extracted_content = content_processor.process_content(source_url)
    serializable_output = _serialize_extracted_content(extracted_content)

    # Assert
    assert serializable_output == golden.out['extracted_contents']  # noqa: Q000

@pytest.mark.golden_test("golden_files/test_content_processor_local_image.yml")
def test_local_multimodal_file_behavior(golden, fs, mock_get_mime_type):  # noqa: ANN001, ANN201, ARG001
    """
    Characterizes processing of a local, non-text file (an image).
    """
    # Arrange
    fake_image_path = "/test_data/image.png"
    fs.create_file(fake_image_path, contents=b'fakepngcontent')  # noqa: Q000
    source_path = golden['input']['source_path']  # noqa: Q000
    content_processor = ContentProcessor()

    # Act
    extracted_content = content_processor.process_content(source_path)
    serializable_output = _serialize_extracted_content(extracted_content)

    # Assert
    assert serializable_output == golden.out['extracted_contents']  # noqa: Q000

@pytest.mark.golden_test("golden_files/test_content_processor_mixed_list.yml")
def test_mixed_content_list_behavior(golden, fs, mock_get_mime_type):  # noqa: ANN001, ANN201, ARG001
    """
    Characterizes processing of a list containing mixed content types.
    """
    # Arrange
    fs.create_file("/test_data/report.txt", contents="Text from a file.")
    mixed_source_list = golden['input']['source_list']  # noqa: Q000
    content_processor = ContentProcessor()

    # Act
    extracted_contents = content_processor.process_content(mixed_source_list)
    serializable_output = _serialize_extracted_content(extracted_contents)

    # Assert
    assert serializable_output == golden.out['extracted_contents']  # noqa: Q000
