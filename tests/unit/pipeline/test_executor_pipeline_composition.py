from gemini_batch.config import GeminiConfig
from gemini_batch.executor import create_executor


def test_default_pipeline_includes_expected_handlers():
    ex = create_executor(GeminiConfig(api_key="k", model="gemini-2.0-flash"))
    names = [
        h.__class__.__name__ for h in ex._pipeline
    ]  # accessing internal for contract check
    # Order should be SourceHandler -> ExecutionPlanner -> RateLimitHandler -> APIHandler -> ResultBuilder
    assert names[:5] == [
        "SourceHandler",
        "ExecutionPlanner",
        "RateLimitHandler",
        "APIHandler",
        "ResultBuilder",
    ]
