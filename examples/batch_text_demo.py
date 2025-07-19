#!/usr/bin/env python3  # noqa: EXE001
"""
Demonstration of batch processing with text content.
"""  # noqa: D200, D212

import time

import requests

from gemini_batch import BatchProcessor
from gemini_batch.response import calculate_quality_score


def get_short_ai_content():  # noqa: ANN201
    """Original short AI content for quick testing"""  # noqa: D415
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


def get_extended_ai_content():  # noqa: ANN201
    """Extended AI content for better batch processing demonstration"""  # noqa: D415
    content = """
    Artificial Intelligence (AI) represents one of the most transformative technologies
    of the 21st century, fundamentally reshaping how we interact with information,
    solve complex problems, and understand the world around us. The field has evolved
    dramatically from its early theoretical foundations in the 1950s to today's
    sophisticated systems that demonstrate remarkable capabilities across multiple domains.

    Modern AI systems excel in natural language processing, enabling machines to
    understand, interpret, and generate human language with unprecedented accuracy.
    These systems can translate between languages, summarize complex documents,
    answer questions, and even engage in creative writing tasks. Computer vision
    has similarly advanced, allowing AI to recognize objects, faces, and patterns
    in images and videos with superhuman precision in many cases.

    Machine learning, the driving force behind most modern AI applications, enables
    systems to learn from data without being explicitly programmed for every task.
    Deep learning, a subset of machine learning using neural networks with multiple
    layers, has been particularly revolutionary. These networks can identify complex
    patterns in vast datasets, leading to breakthroughs in image recognition,
    speech processing, and predictive analytics.

    Key applications span numerous industries. In healthcare, AI assists with medical
    diagnosis by analyzing medical images, predicting disease progression, and
    accelerating drug discovery processes. Radiologists now work alongside AI systems
    that can detect cancer in medical scans with remarkable accuracy. In drug discovery,
    AI algorithms can identify potential drug compounds and predict their efficacy,
    reducing the time and cost of bringing new medications to market.

    The finance sector leverages AI for algorithmic trading, where systems can analyze
    market patterns and execute trades in milliseconds. Fraud detection systems use
    machine learning to identify suspicious transaction patterns, protecting consumers
    and financial institutions from financial crimes. Credit scoring and risk assessment
    have been revolutionized by AI's ability to analyze vast amounts of financial data.

    Transportation is being transformed through autonomous vehicles, which use AI
    to navigate roads, avoid obstacles, and make split-second driving decisions.
    These systems combine computer vision, sensor fusion, and predictive algorithms
    to create safer and more efficient transportation systems.

    In entertainment and media, AI powers recommendation systems that suggest content
    based on user preferences and behavior patterns. Computer graphics and animation
    increasingly rely on AI for realistic rendering, motion capture, and even
    generating synthetic media content.

    However, the rapid advancement of AI also presents significant challenges that
    society must address. Bias in AI systems is a critical concern, as these systems
    can perpetuate or amplify existing societal biases present in their training data.
    This can lead to unfair treatment in hiring, lending, criminal justice, and other
    critical decision-making processes.

    Job displacement concerns arise as AI systems become capable of performing tasks
    traditionally done by humans. While AI creates new job categories, it also
    eliminates others, requiring significant workforce retraining and adaptation.
    The economic and social implications of this transition require careful management
    and policy intervention.

    Privacy implications are substantial, as AI systems often require vast amounts
    of personal data to function effectively. The collection, storage, and use of
    this data raise important questions about individual privacy rights and data
    protection regulations.

    The need for transparent and explainable AI decisions becomes critical as these
    systems are deployed in high-stakes environments. Understanding how AI systems
    reach their conclusions is essential for building trust, ensuring accountability,
    and identifying potential errors or biases.

    Ethical considerations extend beyond technical challenges to fundamental questions
    about the role of AI in society. Issues such as autonomous weapons, surveillance
    systems, and the concentration of AI capabilities in the hands of a few large
    corporations require ongoing dialogue between technologists, policymakers, and
    society at large.

    Looking forward, the future of AI promises even more transformative changes.
    Advances in quantum computing may unlock new AI capabilities, while improvements
    in energy efficiency could make AI more accessible and sustainable. The integration
    of AI with other emerging technologies like biotechnology, nanotechnology, and
    robotics will likely create new possibilities and challenges.

    As AI continues to evolve, the importance of responsible development and deployment
    cannot be overstated. This includes ensuring AI systems are safe, reliable,
    fair, and aligned with human values. International cooperation on AI governance,
    standards, and safety protocols will be essential for maximizing the benefits
    while minimizing the risks of this powerful technology.
    """  # noqa: E501

    questions = [
        "What are the main technical capabilities that modern AI systems demonstrate?",
        "How has machine learning, particularly deep learning, revolutionized AI?",
        "What specific applications of AI in healthcare show the most promise?",
        "How does AI impact financial services beyond just trading algorithms?",
        "What are the primary ethical and societal challenges posed by AI advancement?",
        "What role does data privacy play in AI system development and deployment?",
        "How might emerging technologies like quantum computing affect AI's future?",
        "What approaches are needed for responsible AI development and governance?",
    ]

    return content.strip(), questions


def get_public_domain_content():  # noqa: ANN201
    """Fetch content from a public domain source for demonstration"""  # noqa: D415
    try:
        print("📥 Attempting to fetch Pride and Prejudice from Project Gutenberg...")
        # Example: Fetch a short public domain text
        # This is just an example - you could use any public API or text source
        response = requests.get(
            "https://www.gutenberg.org/files/1342/1342-0.txt",
            timeout=10,
        )
        if response.status_code == 200:  # noqa: PLR2004
            print("✅ Successfully fetched content from Project Gutenberg")
            # Take a substantial excerpt from the beginning of the book
            full_text = response.text
            start_pos = full_text.find("Chapter I")
            if start_pos != -1:
                # Extract first few chapters (≈ 3,000-4,000 words)
                content = full_text[start_pos : start_pos + 15000]
                print(
                    f"✅ Extracted {len(content):,} characters starting from Chapter I"  # noqa: COM812
                )

                questions = [
                    "What are the main themes presented in these opening chapters?",
                    "How does the author establish the social context of the story?",
                    "What can we infer about the historical period from the text?",
                    "How does the narrative style contribute to character development?",
                    "What social commentary is evident in the author's portrayal "
                    "of family dynamics?",
                ]

                return content.strip(), questions
            else:  # noqa: RET505
                print("Could not find Chapter I in downloaded content")
        else:
            print(f"HTTP {response.status_code} error fetching content")
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching public domain content: {e}")

    # Fallback to extended content if download fails
    print("📋 Falling back to extended AI content...")
    return get_extended_ai_content()


def run_demo_with_content(name, content, questions, processor, compare_methods=True):  # noqa: ANN001, ANN201, FBT002
    """Run demo with specific content and display results"""  # noqa: D415
    print(f"\n{'=' * 60}")
    print(f"🧪 TESTING: {name}")
    print(f"Content length: {len(content):,} characters")
    print(f"Questions: {len(questions)}")
    print(f"{'=' * 60}")

    # Process the questions
    results = processor.process_questions(
        content, questions, compare_methods=compare_methods  # noqa: COM812
    )

    # Display truncated answers
    print("\n📝 ANSWERS (Preview):")
    print("-" * 40)
    for i, answer in enumerate(results["answers"], 1):
        preview = answer[:80] + "..." if len(answer) > 80 else answer  # noqa: PLR2004
        print(f"Q{i}: {preview}")

    # Display efficiency metrics
    efficiency = results["efficiency"]
    print("\n📊 TOKEN ECONOMICS ANALYSIS:")
    print("-" * 40)

    if efficiency["comparison_available"]:
        individual_total = results["metrics"]["individual"]["total_tokens"]
        batch_total = results["metrics"]["batch"]["total_tokens"]
        tokens_saved = individual_total - batch_total
        savings_percent = (
            (tokens_saved / individual_total) * 100 if individual_total > 0 else 0
        )

        print(f"📞 Individual calls: {results['metrics']['individual']['calls']}")
        print(f"📦 Batch calls: {results['metrics']['batch']['calls']}")
        print(f"🔢 Tokens - Individual: {individual_total:,}")
        print(f"🔢 Tokens - Batch: {batch_total:,}")
        print(f"💰 Tokens saved: {tokens_saved:,} ({savings_percent:.1f}% reduction)")
        print(f"⚡ Efficiency improvement: {efficiency['token_efficiency_ratio']:.1f}x")
        print(f"🚀 Time improvement: {efficiency['time_efficiency']:.1f}x")
        print(
            f"🎯 Meets 3x target: {'✅ Yes' if efficiency['meets_target'] else '❌ No'}"  # noqa: COM812
        )

        # Calculate and display quality score
        quality_score = calculate_quality_score(
            results.get("individual_answers", []), results["answers"]  # noqa: COM812
        )
        if quality_score is not None:
            print(f"✨ Quality score: {quality_score * 100:.1f}/100")
    else:
        print("Batch processing completed successfully")
        print("Individual comparison not available")

    return results


def main():  # noqa: ANN201, D103
    print("Gemini Batch Processing Framework - Enhanced Demo")
    print("Testing different content lengths to demonstrate efficiency gains")

    processor = BatchProcessor()

    # Option to run batch-only mode for faster demos
    batch_only = False

    # Test with different content lengths
    test_cases = [
        ("Extended AI Content", *get_extended_ai_content()),
        ("Pride and Prejudice", *get_public_domain_content()),
    ]

    results_summary = []

    for i, (name, content, questions) in enumerate(test_cases):
        try:
            # Add delay between test cases to respect rate limits
            if i > 0:
                print("\n⏳ Waiting 5 seconds to respect rate limits...")
                time.sleep(5)

            results = run_demo_with_content(
                name, content, questions, processor, compare_methods=not batch_only  # noqa: COM812
            )
            efficiency = results["efficiency"]

            if efficiency["comparison_available"]:
                results_summary.append(
                    {
                        "name": name,
                        "content_length": len(content),
                        "questions": len(questions),
                        "efficiency_ratio": efficiency["token_efficiency_ratio"],
                        "meets_target": efficiency["meets_target"],
                        "tokens_saved": (
                            results["metrics"]["individual"]["total_tokens"]
                            - results["metrics"]["batch"]["total_tokens"]
                        ),
                    }  # noqa: COM812
                )
        except Exception as e:  # noqa: BLE001
            print(f"Error testing {name}: {e}")

    # Summary comparison
    if results_summary:
        print(f"\n{'=' * 60}")
        print("📊 SUMMARY COMPARISON")
        print(f"{'=' * 60}")
        print(f"{'Test Case':<25} {'Length':<10} {'Efficiency':<12} {'Target Met':<12}")
        print("-" * 60)

        for result in results_summary:
            print(
                f"{result['name']:<25} "
                f"{result['content_length']:,<10} "
                f"{result['efficiency_ratio']:.1f}x{'':<8} "
                f"{'✅ Yes' if result['meets_target'] else '❌ No':<12}"  # noqa: COM812
            )

    print(
        "\n💡 KEY INSIGHT: Longer content demonstrates "
        "greater batch processing benefits"  # noqa: COM812
    )
    print(
        "   because the same content is reused across "
        "multiple questions, reducing redundancy."  # noqa: COM812
    )


if __name__ == "__main__":
    main()
