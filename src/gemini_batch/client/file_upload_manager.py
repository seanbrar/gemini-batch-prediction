"""
File upload management for Gemini Files API
"""  # noqa: D200, D212, D415

from pathlib import Path
import time
from typing import Any

from google import genai

from ..constants import FILE_POLL_INTERVAL, FILE_PROCESSING_TIMEOUT  # noqa: TID252
from ..exceptions import APIError  # noqa: TID252


class FileUploadManager:
    """Manages file uploads to Gemini Files API with processing status tracking"""  # noqa: D415

    def __init__(self, client: genai.Client):  # noqa: ANN204, D107
        self.client = client

    def upload_and_wait(self, file_path: Path) -> Any:  # noqa: ANN401
        """Upload file and wait for processing completion."""
        try:
            uploaded_file = self.client.files.upload(file=str(file_path))
            self._wait_for_processing(uploaded_file)
            return uploaded_file  # noqa: TRY300
        except Exception as e:
            raise APIError(f"Failed to upload file {file_path}: {e}") from e  # noqa: EM102, TRY003

    def _wait_for_processing(
        self, uploaded_file: Any, timeout: int = FILE_PROCESSING_TIMEOUT  # noqa: ANN401, COM812
    ) -> None:
        """Wait for uploaded file to finish processing."""
        start_time = time.time()
        poll_interval = FILE_POLL_INTERVAL

        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise APIError(f"File processing timeout: {uploaded_file.display_name}")  # noqa: EM102, TRY003

            time.sleep(poll_interval)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise APIError(f"File processing failed: {uploaded_file.display_name}")  # noqa: EM102, TRY003
