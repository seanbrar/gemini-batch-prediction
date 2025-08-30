import pytest

from gemini_batch.core.types import Source

pytestmark = pytest.mark.unit


def test_from_youtube_valid_examples():
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "http://youtube.com/watch?v=abc123",
    ]
    for u in urls:
        s = Source.from_youtube(u)
        assert s.source_type == "youtube"
        assert s.identifier == u
        assert s.mime_type == "video/youtube"
        assert s.size_bytes == 0
        assert s.content_loader() == u.encode("utf-8")


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "not-a-url",
        "https://example.com/watch?v=abc",  # not youtube
        "file:///path",
    ],
)
def test_from_youtube_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        Source.from_youtube(bad)


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("1706.03762", "https://arxiv.org/pdf/1706.03762.pdf"),
        ("1706.03762v5", "https://arxiv.org/pdf/1706.03762v5.pdf"),
        ("cs.CL/9901001", "https://arxiv.org/pdf/cs.CL/9901001.pdf"),
        (
            "https://arxiv.org/abs/1706.03762",
            "https://arxiv.org/pdf/1706.03762.pdf",
        ),
        (
            "https://arxiv.org/abs/1706.03762v3",
            "https://arxiv.org/pdf/1706.03762v3.pdf",
        ),
        (
            "https://arxiv.org/pdf/1706.03762.pdf",
            "https://arxiv.org/pdf/1706.03762.pdf",
        ),
        (
            "https://arxiv.org/pdf/1706.03762v2.pdf",
            "https://arxiv.org/pdf/1706.03762v2.pdf",
        ),
        # tolerate missing .pdf on pdf path
        (
            "https://arxiv.org/pdf/1706.03762",
            "https://arxiv.org/pdf/1706.03762.pdf",
        ),
    ],
)
def test_from_arxiv_normalization(ref: str, expected: str) -> None:
    s = Source.from_arxiv(ref)
    assert s.source_type == "arxiv"
    assert s.identifier == expected
    assert s.mime_type == "application/pdf"
    assert s.size_bytes == 0
    assert s.content_loader() == expected.encode("utf-8")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "170603762",  # not clearly arXiv (missing dot or slash)
        "https://arxiv.org/other/1706.03762",  # unsupported shape
    ],
)
def test_from_arxiv_invalid_inputs(bad: str) -> None:
    with pytest.raises(ValueError):
        Source.from_arxiv(bad)


def test_from_uri_rejects_specialized_cases() -> None:
    with pytest.raises(ValueError):
        Source.from_uri("https://www.youtube.com/watch?v=abc", "video/youtube")
    with pytest.raises(ValueError):
        Source.from_uri("https://arxiv.org/abs/1706.03762", "application/pdf")
