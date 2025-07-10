#!/usr/bin/env python3
"""
Conversation Demo: Document Analysis & Synthesis

Shows how to use conversation sessions for analyzing and synthesizing
information across multiple documents with natural follow-up questions.
"""

from gemini_batch import create_conversation


def main():
    print("📄 Document Analysis & Synthesis Demo")
    print("=" * 45)

    # Analyze multiple research papers together
    papers_directory = "examples/test_data/research_papers/"
    session = create_conversation(papers_directory)

    print(f"📚 Analyzing papers from: {papers_directory}")

    # Initial analysis questions
    analysis_questions = [
        "What are the main research themes across these papers?",
        "Which papers discuss efficiency techniques?",
        "What are the key methodological approaches used?",
    ]

    print("\n🔍 Performing initial analysis...")
    answers = session.ask_multiple(analysis_questions)

    for i, (q, a) in enumerate(zip(analysis_questions, answers), 1):
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {a[:100]}...")

    # Follow-up synthesis question
    print("\n🧠 Synthesizing findings...")
    synthesis = session.ask(
        "Based on the themes and methodologies you identified, what would be the most promising research direction for future work?"
    )
    print(f"Synthesis: {synthesis[:200]}...")

    # Show conversation context at work
    print("\n🔗 Testing conversation memory...")
    memory_test = session.ask(
        "Which specific paper would be most relevant to the research direction you just recommended?"
    )
    print(f"Context-aware response: {memory_test[:150]}...")

    # Session summary
    stats = session.get_stats()
    print("\n📊 Analysis Summary:")
    print(f"   Total questions: {stats['total_turns']}")
    print(f"   Sources analyzed: {stats['active_sources']}")
    print(f"   Processing time: {stats['session_duration']:.1f} seconds")

if __name__ == "__main__":
    main()
