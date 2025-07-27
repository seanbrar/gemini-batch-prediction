from pydantic import BaseModel, Field


class MockCommit(BaseModel):
    """A Pydantic model to represent a commit for testing."""

    type: str = "feat"  # Default type
    scope: str
    descriptions: list[str]
    short_hash: str
    hexsha: str
    breaking_descriptions: list[str] = Field(default_factory=list)
    linked_issues: list[str] = Field(default_factory=list)

    def __lt__(self, other: "MockCommit") -> bool:
        """Provides a default sort order for stable sorting in Jinja."""
        if not isinstance(other, MockCommit):
            return NotImplemented
        return self.short_hash < other.short_hash
