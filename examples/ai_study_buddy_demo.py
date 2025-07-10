#!/usr/bin/env python3
"""
AI Study Buddy Demo: Learning Deep Learning Fundamentals

Shows how conversation sessions enable natural learning progression:
- Start with foundational concepts from papers
- Add video content for practical understanding
- Build knowledge through contextual follow-ups
- Demonstrate session persistence and analytics
"""

from gemini_batch import create_conversation, load_conversation


def main():
    print("🎓 AI Study Buddy: Learning Deep Learning Fundamentals")
    print("=" * 60)

    # Start learning journey with foundational paper
    print("📚 Starting with foundational concepts...")
    session = create_conversation(
        "examples/test_data/research_papers/lecun_deep_learning_ai_2021.pdf"
    )

    # Build foundational understanding with focused questions
    foundation_questions = [
        "What are the main types of machine learning?",
        "What are the key advantages of neural networks?",
    ]

    answers = session.ask_multiple(foundation_questions)
    print(f"✅ Learned {len(answers)} foundational concepts")

    # Natural follow-up based on previous answers
    print("\n🤔 Following up on the concepts...")
    followup = session.ask(
        "Which advantage would be most important for image recognition?"
    )
    print(f"💡 Insight: {followup[:120]}...")

    # Add video content for practical perspective (shorter video)
    print("\n🎥 Adding video content for practical understanding...")
    session.add_source(
        "https://www.youtube.com/watch?v=aircAruvnKk"
    )  # Use any educational ML video URL

    # Cross-source learning with focused questions
    practical_questions = [
        "How do the theoretical concepts apply to the neural network visualization in the video?"
    ]

    practical_answers = session.ask_multiple(practical_questions)
    print(f"✅ Connected theory to practice with {len(practical_answers)} insights")

    # Add efficiency-focused paper for deeper understanding
    print("\n📄 Adding efficiency research for advanced concepts...")
    session.add_source(
        "examples/test_data/research_papers/menghani_efficient_deep_learning_2021.pdf"
    )

    # Focused synthesis question
    synthesis = session.ask(
        "What would be the most practical first step for someone starting with deep learning?"
    )
    print(f"🧠 Synthesis: {synthesis[:150]}...")

    # Demonstrate session persistence
    print("\n💾 Saving learning session...")
    session_id = session.save()
    print(f"Session saved: {session_id}")

    # Show learning analytics
    stats = session.get_stats()
    print("\n📊 Learning Session Summary:")
    print(f"   Questions asked: {stats['total_turns']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Sources used: {stats['active_sources']}")
    print(f"   Learning time: {stats['session_duration']:.1f} seconds")

    # Demonstrate session loading
    print("\n🔄 Loading previous session...")
    loaded_session = load_conversation(session_id)

    # Continue learning with loaded session
    final_question = loaded_session.ask("What should I study next?")
    print(f"📚 Next steps: {final_question[:100]}...")

    print("\n✨ Learning complete! The conversation system:")
    print(f"   • Maintains context across {len(loaded_session.sources)} sources")
    print("   • Enables natural knowledge building")
    print("   • Supports multimedia learning")
    print("   • Persists learning sessions")


if __name__ == "__main__":
    main()
