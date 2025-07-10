#!/usr/bin/env python3
"""
Using Context Caching

Shows how to enable context caching for cost-effective repeated analysis
of large content. Context caching reduces costs when asking multiple
questions about the same content.

This example demonstrates:
- Enabling caching with enable_caching=True
- Processing multiple questions about the same content
- Verifying that caching is active and working
"""

from gemini_batch import BatchProcessor, GeminiClient


def main():
    """Simple example of using context caching for repeated content analysis"""

    print("📄 Context Caching Example")
    print("=" * 30)

    # Create substantial content (needs to be large enough to trigger caching)
    # Context caching requires 4096+ tokens to be effective
    content = (
        """
    Artificial Intelligence and Machine Learning have transformed numerous industries
    over the past decade, representing one of the most significant technological
    advances in human history. Key developments include natural language processing,
    computer vision, automated decision-making systems, and neural network architectures
    that can process and understand complex patterns in data.

    Natural Language Processing (NLP) enables computers to understand, interpret,
    and generate human language with remarkable sophistication. Applications include
    advanced chatbots, real-time translation services, sentiment analysis systems,
    automated content generation, text summarization, question answering systems,
    and voice recognition technologies. Modern NLP models can understand context,
    nuance, and even generate creative content like poetry and stories.

    Computer Vision allows machines to process and analyze visual information
    with superhuman accuracy in many domains. Use cases span from medical imaging
    and diagnostic systems, autonomous vehicles and transportation systems,
    security and surveillance systems, retail analytics and inventory management,
    quality control in manufacturing, facial recognition systems, and augmented
    reality applications. Advanced computer vision can now recognize objects,
    understand scenes, and even generate new images from text descriptions.

    Machine Learning algorithms learn patterns from data to make predictions or
    decisions without being explicitly programmed for every scenario. Common
    algorithms include neural networks with various architectures, decision trees
    and random forests, support vector machines, clustering algorithms, and
    reinforcement learning systems. Deep learning, a subset of machine learning,
    uses neural networks with multiple layers to model complex patterns and
    has been particularly successful in image recognition, speech processing,
    and natural language understanding.

    The impact of AI extends across numerous sectors including healthcare where
    AI assists in diagnosis, drug discovery, and treatment planning; finance
    where it enables algorithmic trading, fraud detection, and risk assessment;
    education where it provides personalized learning experiences; entertainment
    where it powers recommendation systems and content creation; and scientific
    research where it accelerates discovery and analysis across fields from
    astronomy to biology. The economic implications are vast, with AI expected
    to contribute trillions of dollars to global GDP in the coming decades.
    """
        * 10
    )  # Repeat to ensure sufficient content for caching (needs 4096+ tokens)

    # Multiple questions about the same content
    questions = [
        "What are the main AI technologies mentioned?",
        "What are some applications of NLP?",
        "How is computer vision being used?",
        "What makes machine learning different from traditional programming?",
    ]

    # Create client with caching enabled
    print("🔧 Creating client with caching enabled...")
    client = GeminiClient.from_env(enable_caching=True)
    processor = BatchProcessor(client=client)

    # Process questions (cache will be created automatically)
    print("⚡ Processing questions...")
    results = processor.process_questions(
        content,
        questions,
        return_usage=True,  # Get usage metrics to verify caching
    )

    # Display results
    print(f"\n📝 Generated {len(results['answers'])} answers:")
    for i, answer in enumerate(results["answers"], 1):
        print(f"\n{i}. {answer[:150]}{'...' if len(answer) > 150 else ''}")

    # Verify caching is working
    print("\n🔍 Cache Status:")
    cache_summary = results.get("cache_summary", {})

    if cache_summary.get("cache_enabled"):
        print("  ✅ Context caching is active")

        # Show cache efficiency
        batch_metrics = results["metrics"]["batch"]
        cached_tokens = batch_metrics.get("cached_tokens", 0)
        if cached_tokens > 0:
            print(f"  📊 Cached tokens: {cached_tokens:,}")
            print(
                "  💡 Subsequent calls with same content will be much faster/cheaper!"
            )
        else:
            print("  📊 Cache created (future calls will benefit)")

    else:
        print("  ⚠️  Caching not active (content may be below threshold)")

    print(f"\n✨ Total tokens used: {results['metrics']['batch']['total_tokens']:,}")


if __name__ == "__main__":
    main()
