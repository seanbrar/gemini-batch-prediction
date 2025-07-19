#!/usr/bin/env python3  # noqa: EXE001
"""
Academic Research Demo: Multi-Source Literature Analysis

Demonstrates analyzing multiple research sources (papers, videos, web content)
simultaneously using structured output and cross-source synthesis.

This example shows how to:
- Process mixed content types in a single batch
- Use structured output for systematic analysis
- Perform cross-source synthesis
- Analyze source composition and efficiency

Adapt this template for your own research questions and sources.
"""  # noqa: D212, D415

from typing import List  # noqa: UP035

from pydantic import BaseModel

from gemini_batch import BatchProcessor
from gemini_batch.analysis import ContentAnalyzer


class ResearchSynthesis(BaseModel):
    """Structured output schema for research synthesis"""  # noqa: D415

    executive_summary: str
    main_techniques: List[str]  # noqa: UP006
    most_promising_approach: str
    research_gaps: List[str]  # noqa: UP006
    practical_recommendations: List[str]  # noqa: UP006


def analyze_ai_efficiency_research():  # noqa: ANN201
    """Example: Analyze AI model efficiency research across multiple sources"""  # noqa: D202, D415

    # Define research sources (mix of local files, URLs, videos)
    sources = [
        "../examples/test_data/research_papers/",  # Directory of papers
        "https://arxiv.org/pdf/1503.02531",  # Hinton et al. - Knowledge Distillation
        # Song Han - Efficient AI Computing
        "https://www.youtube.com/watch?v=u1_K4UeAl-s",
    ]

    # Cross-source synthesis questions
    research_questions = [
        "What are the main efficiency techniques proposed across all sources?",
        "Which approaches show the most promising results based on collective "
        "evidence?",
        "What gaps exist in current AI efficiency research based on these sources?",
        "What practical recommendations emerge for practitioners wanting to improve "
        "model efficiency?",
        "Write a concise executive summary synthesizing key findings across all "
        "sources.",
    ]

    # Analyze source composition
    analyzer = ContentAnalyzer()
    content_summary = analyzer.analyze_sources(sources, research_questions)

    print("📊 RESEARCH SCOPE:")
    print(f"Total sources: {content_summary.total_count}")
    print(f"Traditional approach: {content_summary.traditional_api_calls} API calls")
    print(f"Batch approach: {content_summary.batch_api_calls} API call")
    print(f"Efficiency potential: {content_summary.efficiency_factor:.1f}× improvement")  # noqa: RUF001
    print()

    # Process all sources with structured output
    processor = BatchProcessor()

    print("🔄 Processing research sources...")
    result = processor.process_questions_multi_source(
        sources=sources,
        questions=research_questions,
        response_schema=ResearchSynthesis,
        system_instruction=(
            "Provide focused, complete responses. For lists, include 3-5 key items."
        ),
    )

    # Extract structured results
    synthesis: ResearchSynthesis = result["structured_data"]
    processing_time = result.get("processing_time", 0)

    print(
        f"✅ Analyzed {content_summary.total_count} sources in "
        f"{processing_time:.1f} seconds"  # noqa: COM812
    )
    print()

    # Display structured results
    print("🎯 RESEARCH SYNTHESIS:")
    print(f"Executive Summary: {synthesis.executive_summary[:100]}...")
    print(f"Main Techniques: {', '.join(synthesis.main_techniques[:3])}...")
    print(f"Most Promising: {synthesis.most_promising_approach[:80]}...")
    print(f"Research Gaps: {len(synthesis.research_gaps)} identified")
    print(f"Recommendations: {len(synthesis.practical_recommendations)} provided")

    return synthesis


def custom_research_template():  # noqa: ANN201
    """Template for adapting to your own research questions"""  # noqa: D202, D415

    # TODO: Replace with your sources  # noqa: FIX002, TD002, TD003
    your_sources = [
        "path/to/your/papers/",
        "https://your-research-url.com",
        # Add more sources...
    ]

    # TODO: Replace with your research questions  # noqa: FIX002, TD002, TD003
    your_questions = [
        "What are the main themes in this literature?",
        "What methodologies are most common?",
        # Add more questions...
    ]

    # TODO: Define your output schema  # noqa: FIX002, TD002, TD003
    class YourResearchSchema(BaseModel):
        summary: str
        key_findings: List[str]  # noqa: UP006
        # Add more fields...

    # Process (same pattern as above)
    processor = BatchProcessor()
    result = processor.process_questions_multi_source(
        sources=your_sources,
        questions=your_questions,
        response_schema=YourResearchSchema,
    )

    return result["structured_data"]


def main():  # noqa: ANN201, D103
    print("🔬 Academic Research Demo: Multi-Source Literature Analysis")
    print("=" * 60)
    print()

    # Run the AI efficiency research example
    try:
        synthesis = analyze_ai_efficiency_research()

        # Demonstrate structured output access
        print(
            f"📋 Generated {len(synthesis.main_techniques)} techniques, "
            f"{len(synthesis.research_gaps)} gaps identified"  # noqa: COM812
        )

        print()
        print("✨ Success! This demonstrates:")
        print("   • Multi-source processing (papers + web + video)")
        print("   • Structured output for systematic analysis")
        print("   • Cross-source synthesis capabilities")
        print("   • Efficient batch processing workflow")
        print()
        print("📝 Adapt the custom_research_template() function for your own research!")

    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Demo requires network access and API credits: {e}")
        print(
            "   See the custom_research_template() function for the workflow pattern."  # noqa: COM812
        )


if __name__ == "__main__":
    main()
