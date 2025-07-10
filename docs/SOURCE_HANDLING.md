# Source Handling Guide

The `gemini_batch` package can process multiple types of content sources efficiently in a single API call.

## Supported Source Types

```python
from gemini_batch import BatchProcessor

processor = BatchProcessor()

# Text content
text_source = "Your text content here"

# Single file
file_source = "document.pdf"

# Multiple files
files_source = ["file1.txt", "file2.pdf", "file3.docx"]

# Directory (processes all files)
directory_source = "path/to/documents/"

# YouTube URL
youtube_source = "https://youtube.com/watch?v=example"

# Mixed content
mixed_source = ["text content", "file.pdf", "https://youtube.com/watch?v=xyz"]
```

## Multi-Source Processing

### Single Batch Processing

Process multiple distinct source collections in one call:

```python
# Combined analysis: all sources in one API call
sources = [
    "Direct text content",
    "document.pdf", 
    ["file1.txt", "file2.txt"],
    "https://youtube.com/watch?v=example"
]

questions = [
    "What are the main topics?",
    "Compare approaches across all sources"
]

# Enables cross-source comparison and synthesis
result = processor.process_questions_multi_source(sources, questions)
```

### Consecutive Processing

Process multiple sources one at a time:

```python
# Sequential analysis: each source processed individually
sources = [
    "document1.pdf",
    "document2.pdf", 
    "document3.pdf"
]

questions = [
    "What is the main argument?",
    "What evidence is provided?"
]

# Returns separate results for each source
result = processor.process_questions(sources, questions)
```

## Basic Usage Examples

### Single Source Type

```python
# Single document analysis
result = processor.process_questions("document.pdf", questions)
```

### Multiple Files

```python
# Multiple files as one dataset
files = ["report1.pdf", "report2.docx", "data.csv"]
result = processor.process_questions(files, questions)
```

### Mixed Content Types

```python
# Mixed content types in one analysis
content = [
    "Background context text",
    "research_paper.pdf",
    "https://example.com/article",
    "data_directory/"
]
result = processor.process_questions(content, questions)
```

### Directory Processing

```python
# Process entire directory with filtering
processor = BatchProcessor()

# All files in directory
result = processor.process_questions("research_papers/", questions)

# With file type filtering (via file operations)
from gemini_batch.files import scan_directory
pdf_files = scan_directory("documents/", extensions=[".pdf"])
result = processor.process_questions(pdf_files, questions)
```

## Response Structure

All source handling methods return the same structure:

```python
{
    "answers": ["Answer 1", "Answer 2", ...],
    "question_count": 2,
    "source_count": 4,  # For multi-source processing
    "efficiency": {...},
    "usage": {"total_tokens": 1234, ...}
}
```

The system automatically optimizes API calls regardless of source complexity.
