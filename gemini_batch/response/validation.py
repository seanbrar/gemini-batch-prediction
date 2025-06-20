"""
Structured response validation and schema handling
"""

from typing import Any

from .types import ValidationResult


def validate_structured_response(response, expected_schema) -> ValidationResult:
    """Validate a structured response against a schema

    Args:
        response: The API response object
        expected_schema: The expected schema for validation
    """
    raw_text = getattr(response, "text", str(response))

    # Try to validate direct parsed response
    if hasattr(response, "parsed") and response.parsed is not None:
        try:
            validated_data = validate_against_schema(response.parsed, expected_schema)
            return ValidationResult(
                success=True,
                parsed_data=validated_data,
                confidence=1.0,
                validation_method="direct_parsed",
                errors=[],
                raw_text=raw_text,
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                parsed_data=None,
                confidence=0.0,
                validation_method="direct_parsed_failed",
                errors=[f"Schema validation failed: {e}"],
                raw_text=raw_text,
            )
    else:
        return ValidationResult(
            success=False,
            parsed_data=None,
            confidence=0.0,
            validation_method="no_parsed_data",
            errors=["No parsed data available from response"],
            raw_text=raw_text,
        )


def validate_against_schema(data: Any, schema: Any) -> Any:
    """Validate data against Pydantic schema or other schema types"""
    if hasattr(schema, "model_validate"):  # Pydantic model
        if isinstance(data, dict):
            return schema.model_validate(data)
        else:
            # Try to convert to dict
            if hasattr(data, "__dict__"):
                return schema.model_validate(data.__dict__)
            else:
                return schema.model_validate(data)
    elif hasattr(schema, "__origin__"):  # Generic types like List[SomeModel]
        return validate_generic_type(data, schema)
    else:
        # Simple type validation or pass-through
        return data


def validate_generic_type(data: Any, schema: Any) -> Any:
    """Handle List[Model], Dict[str, Model] and other generic types"""
    origin = getattr(schema, "__origin__", None)
    args = getattr(schema, "__args__", ())

    if origin is list and args:
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        item_schema = args[0]
        return [validate_against_schema(item, item_schema) for item in data]

    elif origin is dict and len(args) == 2:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        key_schema, value_schema = args
        return {k: validate_against_schema(v, value_schema) for k, v in data.items()}

    return data
