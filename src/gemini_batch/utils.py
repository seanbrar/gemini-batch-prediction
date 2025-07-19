"""
Core utilities for the Gemini Batch Framework
"""  # noqa: D200, D212, D415

import os
from typing import Any, Dict, List, Optional, Union  # noqa: UP035

from .exceptions import MissingKeyError


def parse_env_bool(key: str, default: bool = False) -> bool:  # noqa: FBT001, FBT002
    """Parse boolean environment variable"""  # noqa: D415
    value = os.getenv(key, "").lower().strip()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_with_fallback(primary_key: str, fallback_key: str) -> Optional[str]:  # noqa: UP045
    """Get environment variable with fallback"""  # noqa: D415
    return os.getenv(primary_key) or os.getenv(fallback_key)


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format"""  # noqa: D415
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")  # noqa: EM101, TRY003

    # Basic length check
    api_key = api_key.strip()
    if len(api_key) < 30:  # Too short  # noqa: PLR2004
        raise MissingKeyError("API key appears to be invalid (too short)")  # noqa: EM101, TRY003

    return True


def validate_inputs(inputs: List[Union[str, Dict[str, Any]]]) -> None:  # noqa: UP006, UP007
    """Validate batch processing inputs"""  # noqa: D415
    if not inputs:
        raise ValueError("Inputs list cannot be empty")  # noqa: EM101, TRY003

    if not isinstance(inputs, list):
        raise ValueError("Inputs must be a list")  # noqa: EM101, TRY003, TRY004

    for i, input_item in enumerate(inputs):
        if not isinstance(input_item, (str, dict)):
            raise ValueError(  # noqa: TRY003, TRY004
                f"Input {i} must be a string or dictionary, got {type(input_item)}"  # noqa: COM812, EM102
            )

        if isinstance(input_item, str) and not input_item.strip():
            raise ValueError(f"Input {i} cannot be empty string")  # noqa: EM102, TRY003

        if isinstance(input_item, dict) and not input_item:
            raise ValueError(f"Input {i} cannot be empty dictionary")  # noqa: EM102, TRY003
