"""
File upload management for Gemini Files API
"""

from pathlib import Path
import time
from typing import Any

from google import genai

from ..constants import FILE_POLL_INTERVAL, FILE_PROCESSING_TIMEOUT
from ..exceptions import APIError


class FileUploadManager:
    """Manages file uploads to Gemini Files API with processing status tracking"""

    def __init__(self, client: genai.Client):
        self.client = client

    def upload_and_wait(self, file_path: Path) -> Any:
        """Upload file and wait for processing completion."""
        try:
            uploaded_file = self.client.files.upload(file=str(file_path))
            self._wait_for_processing(uploaded_file)
            return uploaded_file
        except Exception as e:
            raise APIError(f"Failed to upload file {file_path}: {e}") from e

    def _wait_for_processing(
        self, uploaded_file: Any, timeout: int = FILE_PROCESSING_TIMEOUT
    ) -> None:
        """Wait for uploaded file to finish processing."""
        start_time = time.time()
        poll_interval = FILE_POLL_INTERVAL

        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise APIError(f"File processing timeout: {uploaded_file.display_name}")

            time.sleep(poll_interval)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise APIError(f"File processing failed: {uploaded_file.display_name}")
