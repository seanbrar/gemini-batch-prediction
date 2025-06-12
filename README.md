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
- **Video Analysis**: Support YouTube URLs and uploaded video files

## 🚀 Current Status

**Week 1 (June 2-8, 2025)**: Foundation & Core Text Processing ✅ **COMPLETED**
- ✅ Production-ready API client with authentication and error handling
- ✅ Comprehensive batch processing framework for text content
- ✅ Efficiency tracking utilities with quality analysis
- ✅ Professional documentation and examples
- ✅ **Bonus**: Visualization module for efficiency analysis
- ✅ **Bonus**: Scaling experiments demonstrating 3-6x efficiency gains

**Week 2 (June 9-15, 2025)**: Enhanced Error Handling & Testing 🟡 **IN PROGRESS**
- 🚧 Comprehensive test suite (unit tests, integration tests)
- 🚧 Advanced error handling and retry mechanisms
- 🚧 Performance optimization and rate limiting enhancements
- 🚧 Video processing foundation (planned for Week 3-4)
- 🚧 Advanced caching implementation (planned for Week 4-5)

## 📦 Installation

```bash
git clone https://github.com/seanbrar/gemini-batch-prediction.git
cd gemini-batch-prediction

# Install the package in development mode
pip install -e .

# Optional: Install with visualization dependencies
pip install -e .[viz]

# Optional: Install with development dependencies
pip install -e .[dev]

# Optional: Install with all dependencies
pip install -e .[viz,dev]
```

### Setup API Key
1. Get your API key from [Google AI Studio](https://ai.dev/)
2. Create a `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash  # Optional: defaults to gemini-2.0-flash
```

## 🔥 Quick Start

### Basic Usage
```python
from gemini_batch import GeminiClient

# Initialize client (reads from .env file)
client = GeminiClient()

# Basic content generation
response = client.generate_content("Explain quantum computing in simple terms")
print(response)
```

### Batch Processing (NEW)
```python
from gemini_batch import BatchProcessor

# Initialize batch processor
processor = BatchProcessor()

# Educational content
content = """
Artificial Intelligence represents one of the most transformative technologies
of the 21st century, fundamentally reshaping how we interact with information
and solve complex problems. Modern AI systems demonstrate remarkable capabilities
in natural language processing, computer vision, and creative tasks.
"""

# Multiple related questions
questions = [
    "What makes AI transformative compared to previous technologies?",
    "What are the core capabilities of modern AI systems?",
    "How does AI impact different industries?"
]

# Process efficiently with batch API calls
results = processor.process_text_questions(content, questions, compare_methods=True)

# View efficiency gains
print(f"🚀 Efficiency improvement: {results['efficiency']['token_efficiency_ratio']:.1f}x")
print(f"💰 API calls reduced: {results['metrics']['individual']['calls']} → {results['metrics']['batch']['calls']}")

# Access answers
for i, answer in enumerate(results['batch_answers'], 1):
    print(f"\nQ{i}: {answer[:100]}...")
```

## 🛠️ Development Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1-2 | Text batch processing & API foundation | ✅ **Completed** |
| 2-3 | Error handling, testing & optimization | 🟡 **In Progress** |
| 3-4 | Video processing & YouTube integration | ⚪ Planned |
| 4-5 | Context caching implementation | ⚪ Planned |
| 6-8 | Conversation memory & advanced features | ⚪ Planned |
| 9-11 | Performance optimization & error handling | ⚪ Planned |
| 12-13 | Documentation & final polish | ⚪ Planned |

## 🤝 Contributing

This is a Google Summer of Code project under active development. Feedback and suggestions are welcome! Please open an issue or reach out directly.

## 📄 License

[MIT License](LICENSE) - This project is developed as part of Google Summer of Code 2025.

---

**Note**: This is an active GSoC project. Features and APIs may change as development progresses. Check back weekly for updates!