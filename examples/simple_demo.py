#!/usr/bin/env python3
"""
Basic demonstration of the framework (under development)
"""

from gemini_batch import GeminiClient


def main():
    print("Gemini Batch Processing Framework - Demo")
    print("Week 1 goals: Basic API client and batch processing")

    client = GeminiClient()

    print(
        client.generate_content(
            prompt="Hello, world!", system_instruction="You are a helpful assistant."
        )
    )


if __name__ == "__main__":
    main()
