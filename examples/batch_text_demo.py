#!/usr/bin/env python3
"""
Demonstration of batch processing with text content.
"""

from gemini_batch import BatchProcessor


def create_test_data():
    """Create sample content and questions for testing"""
    content = """
    Artificial Intelligence represents one of the most transformative technologies
    of the 21st century, fundamentally reshaping how we interact with information
    and solve complex problems. Modern AI systems demonstrate remarkable capabilities
    in natural language processing, computer vision, and creative tasks that were
    previously considered uniquely human domains.

    Key applications span healthcare (medical diagnosis, drug discovery), finance
    (algorithmic trading, fraud detection), transportation (autonomous vehicles),
    and entertainment (content recommendation, computer graphics). However, rapid
    AI advancement also presents challenges including bias in systems, job
    displacement concerns, privacy implications, and the need for transparent,
    explainable decisions.
    """

    questions = [
        "What are the core capabilities of modern AI systems?",
        "Which industries benefit most from AI applications?",
        "What are the main challenges AI advancement presents?",
        "How does AI impact healthcare and finance specifically?",
        "What makes AI transformative compared to previous technologies?",
    ]

    return content.strip(), questions


def main():
    print("Gemini Batch Processing Framework - Demo")

    processor = BatchProcessor()

    content, questions = create_test_data()

    results = processor.process_batch(content, questions)

    for i, answer in enumerate(results, 1):
        print(f"Q{i}: {answer[:80]}{'...' if len(answer) > 80 else ''}")


if __name__ == "__main__":
    main()
