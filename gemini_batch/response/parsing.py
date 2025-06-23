"""
Text parsing strategies for extracting structured data from responses
"""

import json
import re

from .types import ParsingResult


def parse_text_with_schema_awareness(text: str, expected_schema) -> ParsingResult:
    """Text parsing with multiple strategies

    Args:
        text: The text to parse
        expected_schema: The expected schema (used for context, not validation)
    """

    # Strategy 1: JSON extraction
    json_result = try_json_extraction(text)
    if json_result.success:
        return json_result

    # Strategy 2: Structured patterns
    pattern_result = try_enhanced_pattern_parsing(text, expected_schema)
    if pattern_result.success:
        return pattern_result

    # TODO: Add Pydantic string extraction if complex schema support needed

    # Strategy 3: Basic fallback
    return ParsingResult(
        success=False,
        parsed_data=None,
        confidence=0.0,
        method="fallback",
        errors=["All parsing strategies failed"],
    )


def try_json_extraction(text: str) -> ParsingResult:
    """Extract JSON from text using multiple patterns"""
    json_patterns = [
        r"```json\s*(.*?)\s*```",  # JSON code blocks
        r"```\s*(\{.*?\})\s*```",  # Generic code blocks with objects
        r"```\s*(\[.*?\])\s*```",  # Generic code blocks with arrays
        r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple object patterns
        r"(\[[^\[\]]*\{[^{}]*\}[^\[\]]*\])",  # Array of objects
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                return ParsingResult(
                    success=True,
                    parsed_data=parsed,
                    confidence=0.9,
                    method="json_extraction",
                    errors=[],
                )
            except json.JSONDecodeError:
                continue

    return ParsingResult(
        success=False,
        parsed_data=None,
        confidence=0.0,
        method="json_extraction",
        errors=["No valid JSON found"],
    )


def try_enhanced_pattern_parsing(text: str, expected_schema) -> ParsingResult:
    """Schema-aware pattern parsing

    Args:
        text: The text to parse
        expected_schema: The expected schema (used for context, not validation)
    """

    # Multi-line key-value extraction
    kv_patterns = [
        r"^([A-Za-z_][A-Za-z0-9_\s]*?):\s*(.*?)(?=\n[A-Za-z_]|\n\n|$)",  # Standard key: value
        r"([A-Za-z_][A-Za-z0-9_\s]*?)=\s*(.*?)(?=\n[A-Za-z_]|\n\n|$)",  # key = value
        r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
    ]

    extracted_data = {}

    for pattern in kv_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            for key, value in matches:
                clean_key = key.strip().lower().replace(" ", "_").replace("-", "_")
                clean_value = value.strip().strip('"').strip("'")

                # Try to parse values as appropriate types
                if clean_value.lower() in ("true", "false"):
                    clean_value = clean_value.lower() == "true"
                elif clean_value.isdigit():
                    clean_value = int(clean_value)
                elif is_float(clean_value):
                    clean_value = float(clean_value)

                extracted_data[clean_key] = clean_value
            break  # Use first successful pattern

    if extracted_data:
        return ParsingResult(
            success=True,
            parsed_data=extracted_data,
            confidence=0.7,
            method="pattern_matching",
            errors=[],
        )

    return ParsingResult(
        success=False,
        parsed_data=None,
        confidence=0.0,
        method="pattern_matching",
        errors=["No structured patterns found"],
    )


def is_float(value: str) -> bool:
    """Check if string represents a float"""
    try:
        float(value)
        return True
    except ValueError:
        return False
