# Gemini Batch Prediction Framework

> **Google Summer of Code 2025 Project**  
> Creating efficient video content analysis through intelligent batching and context caching with the Gemini API

**Organization:** Google DeepMind

---

## 🎯 Project Overview

This project develops a framework for efficiently analyzing educational video content using Google's Gemini API. By implementing intelligent batch processing and context caching strategies, we aim to achieve **4-5x reduction in API calls** while maintaining high-quality responses for educational content analysis.

### Key Goals
- **Batch Processing**: Group related questions to minimize redundant API calls
- **Context Caching**: Leverage Gemini's caching capabilities for repeated analysis
- **Conversation Memory**: Maintain coherent follow-up question handling
- **Multimodal Analysis**: Support text, PDFs, videos, images, and mixed content sources

## 🚀 Current Status

**Week 1 (June 2-8, 2025)**: Foundation & Core Text Processing ✅ **COMPLETED**
- ✅ Production-ready API client with authentication and error handling
- ✅ Comprehensive batch processing framework for text content
- ✅ Efficiency tracking utilities with quality analysis
- ✅ Professional documentation and examples
- ✅ **Bonus**: Visualization module for efficiency analysis
- ✅ **Bonus**: Scaling experiments demonstrating 3-6x efficiency gains

**Week 2 (June 9-15, 2025)**: Enhanced Error Handling & Testing ✅ **COMPLETED**
- ✅ Comprehensive test suite (95%+ coverage across unit and integration tests)
- ✅ Advanced configuration management with flexible API tier support
- ✅ Professional error handling and retry mechanisms
- ✅ Performance optimization and rate limiting enhancements
- ✅ Modern packaging system with optional dependencies
- ✅ Professional documentation and setup guides

**Week 3 (June 16-22, 2025)**: Multimodal Processing & File Handling ✅ **COMPLETED**
- ✅ Unified content processing interface for any source type
- ✅ YouTube URL support with native Gemini integration
- ✅ Files API integration for large content (PDFs, videos, images)
- ✅ Multi-source analysis capabilities (directories, mixed content)
- ✅ **Bonus**: Academic literature review demo with 12 sources
- ✅ **Bonus**: Structured output support with Pydantic schemas

## 📦 Installation

```bash
# Basic installation
pip install -e .

# With visualization capabilities (recommended for demos)
pip install -e .[viz]

# With development dependencies
pip install -e .[dev]

# Full installation
pip install -e .[viz,dev]
```

### API Key Setup
Get your API key from [Google AI Studio](https://ai.dev/) and configure:
```bash
# Create .env file
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash  # Optional: defaults to optimal model for tier
```

See [docs/SETUP.md](docs/SETUP.md) for detailed configuration options.

## 🔥 Quick Start

### Flexible Initialization
```python
from gemini_batch import GeminiClient, BatchProcessor

# Environment-based (recommended)
client = GeminiClient()
processor = BatchProcessor()

# See docs/SETUP.md for advanced configuration options
```

### Unified Content Processing
```python
# Works with any content type - text, files, URLs, directories, or mixed
questions = ["What are the main points?", "What are the key insights?"]

# Text content
results = processor.process_questions("Your text content here", questions)

# YouTube video
results = processor.process_questions("https://youtube.com/watch?v=example", questions)

# PDF file or directory
results = processor.process_questions("document.pdf", questions)
results = processor.process_questions("research_papers/", questions)

# Mixed sources (text + files + URLs)
sources = ["Background context", "paper.pdf", "https://example.com/article"]
results = processor.process_questions(sources, questions)

# Access results
print(f"🚀 Efficiency: {results['efficiency']['token_efficiency_ratio']:.1f}x improvement")
print(f"💰 Token reduction: {results['metrics']['individual']['tokens']} → {results['metrics']['batch']['tokens']}")

for i, answer in enumerate(results['answers'], 1):
    print(f"Q{i}: {answer}")
```

### Multi-Source Research Analysis
```python
# Analyze multiple research sources simultaneously
sources = [
    "research_papers/",           # Directory of papers
    "https://arxiv.org/pdf/...",  # arXiv paper
    "https://youtube.com/watch?v=...",  # Educational video
]

research_questions = [
    "What are the main research trends across all sources?",
    "Which approaches show the most promise?",
    "What gaps exist in current research?"
]

# Single API call processes all sources together
results = processor.process_questions_multi_source(sources, research_questions)
```

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Installation and configuration options
- **[Source Handling](docs/SOURCE_HANDLING.md)** - Working with different content types
- **Examples** - See `examples/` directory for complete demos
- **Notebooks** - Interactive demos in `notebooks/` directory

## 🛠️ Development Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1-2 | Text batch processing & API foundation | ✅ **Completed** |
| 2 | Error handling, testing & configuration | ✅ **Completed** |
| 3 | Multimodal processing & file handling | ✅ **Completed** |
| 4-5 | Context caching implementation | ⚪ **Next** |
| 6-8 | Conversation memory & advanced features | ⚪ Planned |
| 9-11 | Performance optimization & polish | ⚪ Planned |
| 12-13 | Documentation & final delivery | ⚪ Planned |

## 🤝 Contributing

This is a Google Summer of Code project under active development. Feedback and suggestions are welcome! Please open an issue or reach out directly.

## 📄 License

[MIT License](LICENSE) - This project is developed as part of Google Summer of Code 2025.

---

**Note**: This is an active GSoC project. Features and APIs may change as development progresses. Check back weekly for updates!