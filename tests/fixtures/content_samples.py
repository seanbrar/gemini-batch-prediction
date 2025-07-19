"""
Realistic content samples for testing different scenarios.
"""

# Educational content samples
EDUCATIONAL_CONTENT = {
    "short_lesson": """
    Machine learning is a method of data analysis that automates analytical model building.
    It uses algorithms that iteratively learn from data, allowing computers to find hidden
    insights without being explicitly programmed where to look.
    """,  # noqa: E501
    "medium_article": """
    Deep Learning: The Revolutionary Approach to Artificial Intelligence
    
    Deep learning represents a significant breakthrough in artificial intelligence,
    utilizing neural networks with multiple layers to automatically learn and extract
    features from raw data. Unlike traditional machine learning approaches that
    require manual feature engineering, deep learning models can automatically
    discover representations needed for detection or classification from raw data.
    
    The transformer architecture, introduced in the seminal paper "Attention Is All You Need",
    fundamentally changed how we approach sequence modeling problems. By using self-attention
    mechanisms, transformers can process all positions in a sequence simultaneously,
    leading to significant improvements in both training efficiency and model performance.
    
    Applications span computer vision, natural language processing, and speech recognition,
    with systems like GPT and BERT demonstrating human-level performance on many tasks.
    """,  # noqa: E501, W293
    "academic_paper_excerpt": """
    Abstract: This paper presents a comprehensive analysis of attention mechanisms
    in transformer architectures for natural language understanding tasks.
    
    1. Introduction
    Natural language processing has undergone significant transformation with the
    introduction of attention-based models. The self-attention mechanism allows
    models to weigh the importance of different words in a sequence when processing
    each word, leading to more nuanced understanding of context and meaning.
    
    2. Methodology
    We evaluate attention patterns across multiple transformer variants including
    BERT, GPT, and T5 on benchmark datasets including GLUE, SuperGLUE, and SQuAD.
    Our analysis focuses on attention head specialization and layer-wise attention
    evolution during fine-tuning.
    
    3. Results
    Our findings indicate that attention heads develop specialized functions,
    with some focusing on syntactic relationships while others capture semantic
    associations. Lower layers tend to capture positional and syntactic information,
    while higher layers focus on semantic and task-specific patterns.
    """,  # noqa: W293
    "conversation_context": """
    Previous discussion covered the basics of neural networks and backpropagation.
    We established that neural networks learn by adjusting weights based on error
    gradients, and that deep networks can learn hierarchical representations.
    
    Now we'll explore how attention mechanisms improve upon traditional architectures
    by allowing models to focus on relevant parts of the input when making predictions.
    """,  # noqa: W293
}

# Question sets for different testing scenarios
QUESTION_SETS = {
    "basic_comprehension": [
        "What is the main topic discussed?",
        "What are the key benefits mentioned?",
        "What challenges or limitations are described?",
    ],
    "analytical": [
        "Compare the advantages and disadvantages presented.",
        "What evidence supports the main claims?",
        "How do these findings relate to current practices?",
    ],
    "conversational": [
        "Can you explain this in simpler terms?",
        "How does this build on what we discussed earlier?",
        "What would be a good next step to learn more?",
    ],
    "research_focused": [
        "What methodology was used in this research?",
        "What are the key findings and their implications?",
        "What future research directions are suggested?",
        "How does this relate to other work in the field?",
    ],
}

# Multi-source scenarios
MULTI_SOURCE_SCENARIOS = {
    "comparative_analysis": {
        "sources": [
            EDUCATIONAL_CONTENT["medium_article"],
            EDUCATIONAL_CONTENT["academic_paper_excerpt"],
        ],
        "questions": [
            "How do these two perspectives on AI differ?",
            "What common themes emerge across both sources?",
            "Which source provides more technical depth?",
        ],
    },
    "progressive_learning": {
        "sources": [
            EDUCATIONAL_CONTENT["short_lesson"],
            EDUCATIONAL_CONTENT["conversation_context"],
            EDUCATIONAL_CONTENT["medium_article"],
        ],
        "questions": [
            "How does understanding build from basic to advanced concepts?",
            "What prerequisites are needed for the advanced material?",
            "Create a learning pathway based on these materials.",
        ],
    },
}
