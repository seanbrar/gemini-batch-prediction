#!/usr/bin/env python3  # noqa: EXE001
"""
Minimal demonstration of the Gemini Batch Processing Framework

Shows the core interface: ask multiple questions about content in a single call.
This is the simplest possible example - see examples/ for advanced features,
efficiency analysis, and multi-source processing capabilities.

Perfect starting point for understanding how the framework works.
"""  # noqa: D212, D415

from gemini_batch import BatchProcessor


def main():  # noqa: ANN201, D103
    print("🚀 Gemini Batch Processing Framework - MVP Demo\n")

    processor = BatchProcessor()

    content = "Machine learning enables pattern recognition in data."
    questions = ["What is machine learning?", "What does it enable?"]

    results = processor.process_questions(content, questions)

    print("📝 Results:")
    for i, answer in enumerate(results["answers"], 1):
        print(f"{i}. {answer}")

    print("\n✨ Same interface works with files, URLs, videos, and more!")
    print("💡 For large content + multiple questions, try context caching:")
    print(
        "   processor = BatchProcessor(client=GeminiClient.from_env(enable_caching=True))"  # noqa: COM812, E501
    )
    print(
        "\nSee examples/ for caching demos, advanced features, and efficiency analysis."  # noqa: COM812
    )


if __name__ == "__main__":
    main()
