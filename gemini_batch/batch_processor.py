"""
Batch processor for text content analysis
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Tuple, Union
import warnings

from .client import GeminiClient
from .exceptions import (
    APIError,
    MissingKeyError,
    NetworkError,
    UnsupportedContentError,
    ValidationError,
)
from .files import FileOperations
from .utils import extract_answers, track_efficiency


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(
        self, api_key: str = None, client: GeminiClient = None, **client_kwargs
    ):
        """
        Initialize batch processor

        Args:
            api_key: API key for GeminiClient (ignored if client provided)
            client: Pre-configured GeminiClient instance
            **client_kwargs: Additional arguments for GeminiClient
                (model_name, enable_caching)
        """
        try:
            if client is not None:
                if api_key is not None or client_kwargs:
                    warnings.warn(
                        "api_key and client_kwargs ignored when client is provided",
                        stacklevel=2,
                    )
                self.client = client
            else:
                self.client = GeminiClient(api_key=api_key, **client_kwargs)

            # Initialize file operations components
            self.file_ops = FileOperations()

            self.reset_metrics()
        except MissingKeyError:
            raise
        except NetworkError:
            raise

    def reset_metrics(self):
        """Reset tracking metrics"""
        self.individual_calls = 0
        self.batch_calls = 0
        self.individual_prompt_tokens = 0
        self.individual_output_tokens = 0
        self.batch_prompt_tokens = 0
        self.batch_output_tokens = 0
        self.individual_time = 0.0
        self.batch_time = 0.0

    def process_individual(
        self, content: str, questions: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Process questions individually for comparison"""
        answers = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        start_time = time.time()

        for question in questions:
            try:
                response = self.client.generate_content(
                    prompt=f"Content: {content}\n\nQuestion: {question}\n\nAnswer:",
                    return_usage=True,
                )

                answers.append(response["text"])
                usage = response["usage"]
                total_prompt_tokens += usage["prompt_tokens"]
                total_output_tokens += usage["output_tokens"]
                self.individual_calls += 1

            except (APIError, NetworkError) as e:
                answers.append(f"Error: {str(e)}")

        duration = time.time() - start_time
        self.individual_time += duration
        self.individual_prompt_tokens += total_prompt_tokens
        self.individual_output_tokens += total_output_tokens

        return answers, {
            "calls": len(questions),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "tokens": total_prompt_tokens + total_output_tokens,
            "time": duration,
        }

    def process_batch(
        self, content: str, questions: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Process all questions in a single batch call"""
        start_time = time.time()

        try:
            response = self.client.generate_batch(content, questions, return_usage=True)

            # Extract individual answers
            answers = extract_answers(response["text"], len(questions))

            usage = response["usage"]
            duration = time.time() - start_time

            self.batch_calls += 1
            self.batch_prompt_tokens += usage["prompt_tokens"]
            self.batch_output_tokens += usage["output_tokens"]
            self.batch_time += duration

            return answers, {
                "calls": 1,
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["output_tokens"],
                "tokens": usage["total_tokens"],
                "time": duration,
            }

        except (APIError, NetworkError):
            # Fallback to individual processing on batch failure
            return self.process_individual(content, questions)

    def process_questions(
        self,
        content: Union[str, Path, List[Path]],
        questions: List[str],
        compare_methods: bool = False,
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Unified method to process questions about any content type

        Args:
            content: Text string, file path, directory path, or list of file paths
            questions: List of questions to answer
            compare_methods: Whether to compare batch vs individual methods (text only)
            return_usage: Whether to return usage information

        Returns:
            Dictionary with answers and metadata
        """
        if not questions:
            raise ValidationError("Questions are required")

        # Auto-detect content type and process accordingly
        content_type = self._detect_content_type(content)

        if content_type == "text":
            return self._process_text_questions(
                content, questions, compare_methods, return_usage
            )
        elif content_type == "file":
            return self._process_file_questions(content, questions, return_usage)
        elif content_type == "directory":
            return self._process_directory_questions(content, questions, return_usage)
        elif content_type == "multiple_files":
            return self._process_multiple_files_questions(
                content, questions, return_usage
            )
        else:
            raise UnsupportedContentError(f"Unsupported content type: {content_type}")

    def _detect_content_type(self, content: Union[str, Path, List[Path]]) -> str:
        """Detect the type of content being processed"""
        if isinstance(content, list):
            if all(isinstance(item, (str, Path)) for item in content):
                paths = [Path(item) for item in content]
                if all(p.exists() and p.is_file() for p in paths):
                    return "multiple_files"
            raise ValidationError("List content must be all valid file paths")

        if isinstance(content, Path):
            # Path objects are definitely intended as paths
            if content.exists():
                if content.is_file():
                    return "file"
                elif content.is_dir():
                    return "directory"
            raise ValidationError(f"Path does not exist: {content}")

        if isinstance(content, str):
            # For strings, be smarter about detection
            # If it contains newlines or is very long, likely text content
            if "\n" in content or len(content) > 200:
                return "text"

            # Short single-line strings could be paths - check if they exist
            try:
                path = Path(content)
                if path.exists():
                    if path.is_file():
                        return "file"
                    elif path.is_dir():
                        return "directory"
            except (OSError, ValueError):
                # Invalid path characters, etc.
                pass

            # Default to text for strings
            return "text"

        raise ValidationError("Content must be string, Path, or list of Paths")

    def _process_file_questions(
        self,
        file_path: Union[str, Path],
        questions: List[str],
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Process questions about a single file"""
        file_path = Path(file_path)

        try:
            # Extract content
            extracted_content = self.file_ops.extract_content(file_path)

            # Check if this is a multimodal file that requires API upload
            if extracted_content.requires_api_upload:
                # Use Files API for multimodal content
                batch_prompt = self._create_multimodal_batch_prompt(questions)
                result = self.client.generate_content_with_file(
                    file_path, batch_prompt, return_usage=return_usage
                )

                if return_usage:
                    answers = extract_answers(result["text"], len(questions))
                    return {
                        "answers": answers,
                        "file_path": str(file_path),
                        "processing_method": "multimodal_files_api",
                        "usage": result["usage"],
                    }
                else:
                    answers = extract_answers(result, len(questions))
                    return {
                        "answers": answers,
                        "file_path": str(file_path),
                        "processing_method": "multimodal_files_api",
                    }
            else:
                # Use text processing for extracted text content
                result = self._process_text_questions(
                    extracted_content.content,
                    questions,
                    compare_methods=False,
                    return_usage=return_usage,
                )
                result["file_path"] = str(file_path)
                result["processing_method"] = "text_extraction"
                return result

        except Exception as e:
            error_answers = [f"Error processing file: {e}"] * len(questions)
            return {
                "answers": error_answers,
                "file_path": str(file_path),
                "processing_method": "error",
                "error": str(e),
            }

    def _process_directory_questions(
        self,
        directory_path: Union[str, Path],
        questions: List[str],
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Process questions about files in a directory"""
        directory_path = Path(directory_path)

        try:
            # Scan directory for files
            categorized_files = self.file_ops.scan_directory(directory_path)

            # Flatten categorized files into a single list
            all_files = []
            for _, files in categorized_files.items():
                all_files.extend(files)

            if not all_files:
                return {
                    "answers": [f"No processable files found in {directory_path}"]
                    * len(questions),
                    "directory_path": str(directory_path),
                    "processing_method": "directory_scan",
                    "file_count": 0,
                }

            # Process each file
            results = []
            for file_info in all_files:
                file_result = self._process_file_questions(
                    file_info.path, questions, return_usage=False
                )
                results.append(file_result)

            return {
                "directory_results": results,
                "directory_path": str(directory_path),
                "processing_method": "directory_batch",
                "file_count": len(all_files),
                "files_processed": [str(info.path) for info in all_files],
            }

        except Exception as e:
            error_answers = [f"Error processing directory: {e}"] * len(questions)
            return {
                "answers": error_answers,
                "directory_path": str(directory_path),
                "processing_method": "error",
                "error": str(e),
            }

    def _process_multiple_files_questions(
        self,
        file_paths: List[Union[str, Path]],
        questions: List[str],
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Process questions about multiple files"""
        file_paths = [Path(p) for p in file_paths]

        results = []
        for file_path in file_paths:
            file_result = self._process_file_questions(
                file_path, questions, return_usage=False
            )
            results.append(file_result)

        return {
            "file_results": results,
            "processing_method": "multiple_files_batch",
            "file_count": len(file_paths),
            "files_processed": [str(p) for p in file_paths],
        }

    def _create_multimodal_batch_prompt(self, questions: List[str]) -> str:
        """Create batch prompt for multimodal content"""
        prompt = "Please answer each of the following questions about this content:\n\n"

        for i, question in enumerate(questions, 1):
            prompt += f"Question {i}: {question}\n"

        prompt += "\nProvide numbered answers in this exact format:\n"
        for i in range(1, len(questions) + 1):
            prompt += f"Answer {i}: [Your response]\n"

        return prompt

    def _process_text_questions(
        self,
        content: str,
        questions: List[str],
        compare_methods: bool = False,
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Internal method to process multiple questions about text content"""
        if not content or not questions:
            raise ValidationError("Content and questions are required")

        self.reset_metrics()

        # Process batch first
        batch_answers, batch_metrics = self.process_batch(content, questions)

        # Conditionally process individually for comparison
        if compare_methods:
            individual_answers, individual_metrics = self.process_individual(
                content, questions
            )
        else:
            individual_answers = None
            individual_metrics = {
                "calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "time": 0.0,
            }

        # Calculate efficiency metrics
        efficiency = track_efficiency(
            individual_calls=individual_metrics["calls"],
            batch_calls=batch_metrics["calls"],
            individual_prompt_tokens=individual_metrics.get("prompt_tokens", 0),
            individual_output_tokens=individual_metrics.get("output_tokens", 0),
            batch_prompt_tokens=batch_metrics.get("prompt_tokens", 0),
            batch_output_tokens=batch_metrics.get("output_tokens", 0),
            individual_time=individual_metrics["time"],
            batch_time=batch_metrics["time"],
        )

        result = {
            "question_count": len(questions),
            "batch_answers": batch_answers,
            "efficiency": efficiency,
            "metrics": {
                "batch": batch_metrics,
                "individual": individual_metrics,
            },
            # Conditionally include individual answers with dictionary unpacking
            **({"individual_answers": individual_answers} if compare_methods else {}),
        }

        # For consistency with new interface, also include 'answers' key
        result["answers"] = batch_answers

        if return_usage:
            # Convert batch_metrics to standard usage format for consistency
            result["usage"] = {
                "prompt_tokens": batch_metrics.get("prompt_tokens", 0),
                "output_tokens": batch_metrics.get("output_tokens", 0),
                "total_tokens": batch_metrics.get("tokens", 0),
                "cached_tokens": 0,  # Not available from batch metrics
            }

        return result

    # Legacy method - now internal
    def process_text_questions(
        self, content: str, questions: List[str], compare_methods: bool = False
    ) -> Dict[str, Any]:
        """Legacy method - use process_questions() instead"""
        warnings.warn(
            "process_text_questions() is deprecated. Use process_questions() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._process_text_questions(
            content, questions, compare_methods, return_usage=False
        )
