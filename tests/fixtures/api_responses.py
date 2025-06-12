from dataclasses import dataclass


@dataclass
class MockUsageMetadata:
    prompt_token_count: int = 100
    candidates_token_count: int = 50
    cached_content_token_count: int = 0


@dataclass
class MockResponse:
    text: str
    usage_metadata: MockUsageMetadata

    def __post_init__(self):
        # Handle API evolution - add new fields with defaults
        if not hasattr(self.usage_metadata, "total_token_count"):
            self.usage_metadata.total_token_count = (
                self.usage_metadata.prompt_token_count
                + self.usage_metadata.candidates_token_count
            )


# Standard test responses
SAMPLE_RESPONSES = {
    "simple_answer": MockResponse(
        text="Artificial Intelligence is a transformative technology...",
        usage_metadata=MockUsageMetadata(
            prompt_token_count=150, candidates_token_count=75
        ),
    ),
    "batch_answer": MockResponse(
        text="""Answer 1: AI capabilities include natural language processing...
        Answer 2: Machine learning drives modern AI through...""",
        usage_metadata=MockUsageMetadata(
            prompt_token_count=800, candidates_token_count=200
        ),
    ),
    "error_response": Exception("Rate limit exceeded"),
}
