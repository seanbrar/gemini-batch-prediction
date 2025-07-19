from typing import Any  # noqa: D100
import warnings


class SchemaAnalyzer:
    """Analyzes a Pydantic schema for complexity that might cause API errors."""

    # Complexity thresholds (tunable)
    MAX_NESTING_DEPTH = 5
    MAX_PROPERTIES = 50
    WARN_PROPERTY_NAME_LEN = 30

    def analyze(self, schema: Any) -> None:  # noqa: ANN401
        """
        Analyzes the schema and issues warnings if complexity thresholds are exceeded.
        """  # noqa: D200, D212
        if not hasattr(schema, "model_fields"):
            # Not a Pydantic model we can analyze
            return

        warnings_found = []
        nesting_depth = self._get_nesting_depth(schema)
        num_properties = len(schema.model_fields)

        if nesting_depth > self.MAX_NESTING_DEPTH:
            warnings_found.append(
                f"nesting depth of {nesting_depth} (max recommended: {self.MAX_NESTING_DEPTH})"  # noqa: COM812, E501
            )

        if num_properties > self.MAX_PROPERTIES:
            warnings_found.append(
                f"{num_properties} properties (max recommended: {self.MAX_PROPERTIES})"  # noqa: COM812
            )

        for field_name in schema.model_fields:
            if len(field_name) > self.WARN_PROPERTY_NAME_LEN:
                warnings_found.append(f"long property name '{field_name}'")
                break  # Warn only once for long names

        if warnings_found:
            message = (
                f"Schema '{schema.__name__}' is complex and may cause API errors or poor performance due to: "  # noqa: E501
                f"{'; '.join(warnings_found)}. Consider simplifying the schema."
            )
            warnings.warn(message, UserWarning)  # noqa: B028

    def _get_nesting_depth(self, schema: Any, depth: int = 1) -> int:  # noqa: ANN401
        """Recursively calculates the maximum nesting depth of Pydantic models."""
        if depth > self.MAX_NESTING_DEPTH:
            return depth

        max_child_depth = 0
        if hasattr(schema, "model_fields"):
            for field in schema.model_fields.values():
                field_type = field.annotation
                if hasattr(field_type, "model_fields"):  # It's a nested Pydantic model
                    max_child_depth = max(
                        max_child_depth, self._get_nesting_depth(field_type, depth + 1)  # noqa: COM812
                    )

        return max(depth, max_child_depth)
