# Gemini Batch Processing Examples

This directory contains simple, lightweight examples demonstrating how to use the Gemini Batch Processing Framework.

## 🚀 Getting Started

### [`simple_demo.py`](simple_demo.py)

**Perfect starting point** - Shows the basic interface for processing multiple questions about content.

```bash
python simple_demo.py
```

## 💬 Conversation Examples

### [`ai_study_buddy_demo.py`](ai_study_buddy_demo.py)

**AI Study Buddy** - Learn deep learning fundamentals through natural conversation with multiple sources (papers + video).

### [`conversation_demo.py`](conversation_demo.py)

**Document Analysis** - Analyze and synthesize information across multiple research papers.

## 📚 Batch Processing Examples

### [`batch_text_demo.py`](batch_text_demo.py)

Process multiple content sources with efficiency analysis.

### [`academic_research_demo.py`](academic_research_demo.py)

Research paper processing and literature review capabilities with structured output.

### [`using_context_caching.py`](using_context_caching.py)

Demonstrates context caching for cost-effective repeated analysis of large content.

## 🔧 Setup

Make sure your API key is set:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Start with `simple_demo.py` to see the basic functionality, then try the conversation demos for more advanced use cases!

## 💡 Tips

- **Conversation system** maintains context across questions and sources - perfect for learning and research
- **Context caching** is automatically optimized when enabled with `enable_caching=True`
- Works best with larger content (documents, papers, etc.) and multiple questions
- All examples are designed to be lightweight and easy to understand
