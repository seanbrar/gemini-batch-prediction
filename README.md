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

**Week 1 (June 2-8, 2025)**: Foundation & Core Text Processing
- ✅ Basic API client with authentication
- ✅ Simple batch processing for text content
- ✅ Efficiency tracking utilities
- 🚧 Video processing (planned for Week 3-4)
- 🚧 Advanced caching and optimization (planned for Week 4-5)

## 📦 Installation

```bash
git clone https://github.com/seanbrar/gemini-batch-prediction.git
cd gemini-batch-prediction
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
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

## 🛠️ Development Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1-2 | Text batch processing & API foundation | 🟡 In Progress |
| 3-4 | Video processing & YouTube integration | ⚪ Planned |
| 4-5 | Context caching implementation | ⚪ Planned |
| 6-8 | Conversation memory & advanced features | ⚪ Planned |
| 9-11 | Optimization & error handling | ⚪ Planned |
| 12-13 | Documentation & final polish | ⚪ Planned |

## 🤝 Contributing

This is a Google Summer of Code project under active development. Feedback and suggestions are welcome! Please open an issue or reach out directly.

## 📄 License

[MIT License](LICENSE) - This project is developed as part of Google Summer of Code 2025.

---

**Note**: This is an active GSoC project. Features and APIs may change as development progresses. Check back weekly for updates!
