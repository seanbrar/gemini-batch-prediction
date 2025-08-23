"""Source resolution stage of the pipeline."""

from collections.abc import Callable, Iterable
import logging
from pathlib import Path
from typing import Any

from gemini_batch.core.exceptions import SourceError
from gemini_batch.core.types import (
    Failure,
    InitialCommand,
    ResolvedCommand,
    Result,
    Source,
    Success,
)
from gemini_batch.pipeline.base import BaseAsyncHandler

logger = logging.getLogger(__name__)


class SourceHandler(BaseAsyncHandler[InitialCommand, ResolvedCommand, SourceError]):
    """Resolves raw inputs into structured Source objects.

    This handler identifies and validates input sources, whether they are
    files, URLs, or text content. It preserves legacy functionality while
    transforming everything into simple `Source` objects.

    TODO: Consider optional, explicit configuration for:
      - Include/Exclude patterns for directory scanning
      - Emitting neutral capability hints for planners (e.g., suggested upload)
      - Vendor capability matrices in planners/API handlers
    """

    def __init__(self) -> None:  # Keep constructor trivial per simplicity rubric
        """Initialize handler (no heavy dependencies)."""
        # Intentionally empty; handler is stateless and uses stdlib utilities

    async def handle(
        self, command: InitialCommand
    ) -> Result[ResolvedCommand, SourceError]:
        """Resolve sources in a command into immutable `Source` objects."""
        try:
            resolved_sources: list[Source] = []
            for raw_source in command.sources:
                try:
                    resolved_sources.extend(self._resolve_single_source(raw_source))
                except Exception as inner_error:
                    # Fail the whole stage explicitly; make errors data, not thrown
                    return Failure(
                        SourceError(
                            f"Failed to resolve source '{raw_source}': {inner_error}"
                        )
                    )

            return Success(
                ResolvedCommand(
                    initial=command, resolved_sources=tuple(resolved_sources)
                )
            )
        except SourceError as e:
            return Failure(e)
        except Exception as e:  # Defensive guardrail
            return Failure(SourceError(f"Failed to resolve sources: {e}"))

    # ------------------------------
    # Internal helpers (pure, small)
    # ------------------------------
    def _resolve_single_source(self, raw_source: Any) -> list[Source]:
        """Resolve a single raw input into one or more `Source` objects.

        - Text: single `Source(type='text')`
        - YouTube URL: single `Source(type='youtube')`
        - arXiv PDF URL: single `Source(type='arxiv')`
        - File path: single `Source(type='file')`
        - Directory path: one `Source(type='file')` per contained file
        """
        # Treat strings and Paths uniformly
        src_str = str(raw_source)

        # URL handling first (no FS calls)
        if self._is_youtube_url(src_str):
            return [self._build_youtube_source(src_str)]

        # arXiv PDF URL support (pattern-based; no network)
        if self._is_arxiv_pdf_url(src_str):
            return [self._build_arxiv_source(src_str)]

        # Direct text content (heuristic avoids URLs and existing paths)
        if self._is_text_content(src_str, raw_source):
            return [self._build_text_source(src_str)]

        # Path-like handling
        path = Path(src_str)
        if path.exists():
            if path.is_dir():
                return list(self._resolve_directory(path))
            if path.is_file():
                return [self._build_file_source_from_path(path)]

        # Unsupported URL or non-existent path falls back to error
        raise SourceError(f"Unsupported source or path not found: {raw_source}")

    def _resolve_directory(self, directory: Path) -> Iterable[Source]:
        """Expand a directory into file `Source` objects (recursive)."""
        for file_path in self._iter_files(directory):
            yield self._build_file_source_from_path(file_path)

    # -------- Source builders --------
    def _build_text_source(self, text: str) -> Source:
        size = len(text.encode("utf-8"))
        return Source(
            source_type="text",
            identifier=text,
            mime_type="text/plain",
            size_bytes=size,
            content_loader=self._make_text_loader(text),
        )

    def _build_youtube_source(self, url: str) -> Source:
        return Source(
            source_type="youtube",
            identifier=url,
            mime_type="video/youtube",
            size_bytes=0,
            # Lazy loader returns a stable bytes representation (no network)
            content_loader=self._make_text_loader(url),
        )

    def _build_arxiv_source(self, url: str) -> Source:
        # We avoid network calls here; size is unknown → 0
        return Source(
            source_type="arxiv",
            identifier=url,
            mime_type="application/pdf",
            size_bytes=0,
            content_loader=self._make_text_loader(url),
        )

    def _build_file_source_from_path(self, file_path: Path) -> Source:
        mime = self._determine_mime_type(file_path)
        size = file_path.stat().st_size
        return Source(
            source_type="file",
            identifier=file_path,
            mime_type=mime or "application/octet-stream",
            size_bytes=size,
            content_loader=self._make_file_loader(file_path),
        )

    # ------- simple URL/text detection (no legacy deps) -------
    def _is_url(self, text: str) -> bool:
        return text.startswith(("http://", "https://"))

    def _is_youtube_url(self, url: str) -> bool:
        url_lower = url.lower()
        return self._is_url(url_lower) and any(
            pattern in url_lower
            for pattern in (
                "youtube.com/watch?v=",
                "youtu.be/",
                "youtube.com/embed/",
                "youtube.com/v/",
                "youtube.com/v=",
            )
        )

    def _is_arxiv_pdf_url(self, url: str) -> bool:
        url_lower = url.lower()
        return self._is_url(url_lower) and (
            "arxiv.org/pdf/" in url_lower or "export.arxiv.org/pdf/" in url_lower
        )

    def _is_text_content(self, text: str, original_source: Any) -> bool:
        if isinstance(original_source, Path):
            return False

        # Handle empty strings as text (they represent "no content")
        if text == "":
            return True

        # Check for URLs first
        if self._is_url(text):
            return False

        # Check for multi-line text or very long text (likely text content)
        if "\n" in text or len(text) > 200:
            return True

        # Use existing MIME detection to identify file paths
        try:
            p = Path(text)
            # If the path exists, it's definitely not text content
            if p.exists():
                return False

            # If it has path separators and an extension, likely a file path
            has_separator = any(sep in text for sep in ("/", "\\"))
            if has_separator and p.suffix:
                return False

            # For simple filenames with extensions, be less aggressive
            # Only treat as file path if it's a very obvious file extension
            if p.suffix and len(p.parts) == 1:
                # Only reject text interpretation for very obvious binary/document extensions
                obvious_file_extensions = {
                    ".pdf",
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".mp4",
                    ".zip",
                }
                if p.suffix.lower() in obvious_file_extensions:
                    return False
        except Exception:
            logger.warning(f"Failed to resolve source: {text}")

        return True

    # ------- directory scanning and mime detection -------
    def _iter_files(self, directory: Path) -> Iterable[Path]:
        """Yield files under directory with pruning of excluded dirs."""
        import os

        exclude_dirs = {
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "build",
            "dist",
            ".tox",
            ".coverage",
        }
        # Deterministic ordering: sort dirs and files at each level
        for root, dirnames, filenames in os.walk(directory):
            dirnames[:] = [d for d in sorted(dirnames) if d not in exclude_dirs]
            for fname in sorted(filenames):
                path = Path(root) / fname
                try:
                    if path.is_file():
                        yield path
                except OSError:
                    continue

    def _determine_mime_type(self, file_path: Path) -> str | None:
        import mimetypes

        mime, _ = mimetypes.guess_type(str(file_path))
        if mime:
            return mime
        # Minimal sensible fallbacks
        ext = file_path.suffix.lower()
        fallback = {
            # Text/code
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".py": "text/x-python",
            ".csv": "text/csv",
            ".json": "application/json",
            ".xml": "application/xml",
            # Documents/media
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".avi": "video/avi",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
        }
        return fallback.get(ext, "application/octet-stream")

    # -------- loader factories (typed, avoid lambda inference issues) --------
    def _make_text_loader(self, text: str) -> Callable[[], bytes]:
        def loader() -> bytes:
            return text.encode("utf-8")

        return loader

    def _make_file_loader(self, path: Path) -> Callable[[], bytes]:
        def loader() -> bytes:
            return Path(path).read_bytes()

        return loader
